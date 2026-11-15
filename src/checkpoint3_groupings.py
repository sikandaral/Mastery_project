"""Checkpoint 3, Finding 2: do the first-20 popularity results survive other ways of grouping users?

Everything for this finding is in this one file, and one run produces every number, table and figure of the finding.
Same 124,873 users, windows and outcome as CP2 Finding 2 (src/checkpoint2_profiles.py, imported and not changed).
Steps:
  1. Behavior groups: the CP2 k-means groups (k = 3, 4, 5, 12), described in original units (instructor note #4).
  2. Matrix groups: users grouped by the movies they rated after their first 20 ratings (instructor note #5).
     Settings fixed in advance, not tuned: all movies as columns, each user's row scaled to length 1 so rating volume
     does not dominate, TruncatedSVD to 50 dimensions (20 as a check), k-means with the same k, n_init and seed as CP2.
     Ratings 1-20 are excluded because the outcome (first-20 popular share) must not be part of the grouping.
  3. The CP2 decomposition (mix of groups vs within-group change) for every grouping, and the Adjusted Rand Index
     (ARI) between groupings (1 = identical, 0 = unrelated).
  4. Seed check. k-means starts from random points, so the exact groups depend on the seed and on the order of the
     rows. Users are therefore sorted by userId (the order of the cached user table is arbitrary), and every k-means
     is repeated with 4 more seeds to show how far the mix and within terms move.
  5. Internal checks (asserts): group sizes add up to all users, shares add up to 100%, the share-weighted group rates
     give back each period's overall rate, and mix + within = total.

Run after src/checkpoint1.py has built data/processed/eda.duckdb (the seed check makes it slow):
    python src/checkpoint3_groupings.py
Writes:
    reports/checkpoint3_table1.csv     Table 1   behavior groups, k = 4
    reports/checkpoint3_table_a1.csv   Table A1  matrix groups, k = 4
    reports/checkpoint3_table_a2.csv   Table A2  ARI between groupings
    reports/checkpoint3_table_a3.csv   Table A3  stress tests (seeds, 20 dimensions)
    reports/checkpoint3_groupings.json every number behind the tables and the figure
    figures/cp3_groupings.png          Figure 2
The tables are written already rounded, so copy them into the report as they are.
"""
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import checkpoint2_profiles as p

MAIN_DIMS, CHECK_DIMS = 50, 20  # SVD dimensions: the main run and the sensitivity check
warnings.simplefilter("ignore", RuntimeWarning)  # spurious matmul warnings seen in CP2; labels are checked below


def time_label(days):
    """Same cutoffs as the engagement types of checkpoint1.user_table (days between first and last rating)."""
    if days <= 1:
        return "one-sitting"
    if days <= 30:
        return "short-lived"
    if days <= 365:
        return "returning"
    return "long-term"


def volume_label(n):
    """Same breakpoints as the rating-volume bands of CP2 (100 and 300)."""
    return "light-volume" if n < 100 else "moderate-volume" if n < 300 else "heavy-volume"


def popularity_label(share, low, high):
    """Thirds of the later popular share across all users, as in the taste types of Checkpoint 1."""
    return "niche-leaning" if share <= low else "popular-leaning" if share >= high else "mixed-popularity"


def group_table(u, col, low, high):
    """One row per group of the grouping in column `col`. Behavior columns use all users in the group;
    the era columns use the CP2 windows (late v3, early v4)."""
    late = (u["start"] >= p.WINDOW_START) & (u["start"] < p.V4)
    early = (u["start"] >= p.V4) & (u["start"] < p.WINDOW_END)
    g = u.groupby(col)
    t = pd.DataFrame({
        "users": g.size(),
        "median_ratings": g["n"].median(),
        "median_days_active": g["tenure_days"].median(),
        "median_first_sitting": g["first_session_n"].median(),
        "later_popular_pct": 100 * g["later_pop"].mean(),
        "mean_rating": g["mean_rating"].mean(),
        "pct_of_late_v3": 100 * u[late].groupby(col).size() / late.sum(),
        "pct_of_early_v4": 100 * u[early].groupby(col).size() / early.sum(),
        "first20_popular_late_v3": 100 * u[late].groupby(col)[p.OUTCOME].mean(),
        "first20_popular_early_v4": 100 * u[early].groupby(col)[p.OUTCOME].mean(),
    })
    # Internal checks: sizes and shares add up, and the share-weighted group rates give back each period's overall rate
    assert t["users"].sum() == len(u)
    for share, rate, mask in [("pct_of_late_v3", "first20_popular_late_v3", late),
                              ("pct_of_early_v4", "first20_popular_early_v4", early)]:
        assert abs(t[share].sum() - 100) < 1e-6
        assert abs((t[share].fillna(0) * t[rate].fillna(0)).sum() / 100 - 100 * u[mask][p.OUTCOME].mean()) < 1e-6
    t["description"] = [
        f"{time_label(d)}, {volume_label(n)}, {popularity_label(s / 100, low, high)}".capitalize()
        for d, n, s in zip(t["median_days_active"], t["median_ratings"], t["later_popular_pct"])
    ]
    return t


def report_table(t):
    """Display-ready table for the report (strings formatted from the numbers above)."""
    pair = lambda a, b: t[a].map("{:.1f}".format) + " / " + t[b].map("{:.1f}".format)
    return pd.DataFrame({
        "Group": t.index,
        "Description": t["description"].to_numpy(),
        "Users": t["users"].map("{:,}".format).to_numpy(),
        "Median ratings": t["median_ratings"].map("{:g}".format).to_numpy(),
        "Median days active": t["median_days_active"].map("{:.1f}".format).to_numpy(),
        "Later popular share (%)": t["later_popular_pct"].map("{:.1f}".format).to_numpy(),
        "Users, late v3 / early v4 (%)": pair("pct_of_late_v3", "pct_of_early_v4").to_numpy(),
        "First-20 popular share, late v3 / early v4 (%)":
            pair("first20_popular_late_v3", "first20_popular_early_v4").to_numpy(),
    })


def movie_matrix(u):
    """User x movie 0/1 matrix from ratings 21 onward (rows in the order of u), each row scaled to length 1."""
    con = duckdb.connect(str(p.POP_DB), read_only=True)
    con.register("users", u[["userId"]])
    r = con.execute("select s.userId, s.movieId from seqp s join users using (userId) where s.ord > 20").df()
    con.close()

    row = pd.Index(u["userId"]).get_indexer(r["userId"])  # row number of each rating's user (same order as u)
    X = csr_matrix((np.ones(len(r), dtype="float32"), (row, r["movieId"].to_numpy())),
                   shape=(len(u), r["movieId"].max() + 1))
    X = normalize(X)  # every user's row gets length 1, so heavy raters do not dominate
    print(f"Matrix: {X.shape[0]:,} users, {len(r):,} ratings (ratings 21+)")
    return X


def matrix_groups(X, dims):
    """Compress X to `dims` numbers per user, then k-means for every k.
    Returns the labels (a dict by k), the compressed matrix Z and the share of variance the compression keeps."""
    svd = TruncatedSVD(n_components=dims, random_state=0)
    Z = svd.fit_transform(X)
    assert np.isfinite(Z).all()
    labels = {}
    for k in p.KS:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
        assert p.check_assignment(Z, km.cluster_centers_, km.labels_) > 0.99  # independent check of the labels
        labels[k] = km.labels_
    return labels, Z, float(svd.explained_variance_ratio_.sum())


def decompose_labels(u, labels):
    """The CP2 decomposition for a grouping given as an array of labels (one per row of u)."""
    u["tmp_group"] = labels
    d = p.decompose_cells(u, ["tmp_group"])
    u.drop(columns="tmp_group", inplace=True)
    return d


def seed_check(u, Xb, Z, dec):
    """Repeat every k-means with 4 more seeds (the main run is seed 0) and recompute the decomposition.
    Xb = standardized behavior features (as in p.cluster), Z = compressed movie matrix, dec = the main decomposition.
    Returns one row per grouping: lowest and highest mix and within term over the 5 runs, and the lowest ARI
    between a repeat's groups and the main groups (1 = the same groups every time)."""
    rows = []
    for name, X, col, key in [("behavior", Xb, "k", "k-means, k = {}"), ("matrix", Z, "mx", "matrix, k = {}")]:
        for k in p.KS:
            main = u[f"{col}{k}"].to_numpy()
            mix = [float(dec.loc[key.format(k), "composition (pp)"])]
            within = [float(dec.loc[key.format(k), "within (pp)"])]
            ari = []
            for seed in range(1, 5):
                km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
                assert p.check_assignment(X, km.cluster_centers_, km.labels_) > 0.99
                d = decompose_labels(u, km.labels_)
                mix.append(d["composition (pp)"])
                within.append(d["within (pp)"])
                ari.append(adjusted_rand_score(main, km.labels_))
            rows.append({"grouping": f"{name}, k = {k}", "mix min": min(mix), "mix max": max(mix),
                         "within min": min(within), "within max": max(within), "lowest ARI vs main": min(ari)})
            print(f"  seed check done: {name}, k = {k}")
    return pd.DataFrame(rows).set_index("grouping")


def ari_table(ari):
    """Table A2: ARI between groupings, two decimals."""
    t = ari.apply(lambda c: c.map(lambda x: f"{round(x, 2) + 0.0:.2f}"))  # + 0.0 avoids "-0.00"
    t.columns = ["Matrix vs behavior", "Matrix vs rule-based", "Behavior vs rule-based"]
    t.insert(0, "k", [i.replace("k = ", "") for i in t.index])
    return t.reset_index(drop=True)


def stress_table(dec, seeds, dec20):
    """Table A3: main run, range over 5 seeds, lowest ARI, and the 20-dimension result (matrix groups only)."""
    fmt = lambda x: f"{round(float(x), 2) + 0.0:.2f}"  # (+ 0.0 avoids "-0.00")
    rng = lambda lo, hi: fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)} to {fmt(hi)}"
    rows = []
    for name, key in [("Behavior", "k-means, k = {}"), ("Matrix", "matrix, k = {}")]:
        for k in p.KS:
            s = seeds.loc[f"{name.lower()}, k = {k}"]
            main = dec.loc[key.format(k)]
            d20 = f"{fmt(dec20.loc[key.format(k), 'composition (pp)'])} / {fmt(dec20.loc[key.format(k), 'within (pp)'])}" \
                if name == "Matrix" else "n/a"
            rows.append({"Grouping": f"{name}, k = {k}",
                         "Mix, main run": fmt(main["composition (pp)"]),
                         "Mix, 5 seeds": rng(s["mix min"], s["mix max"]),
                         "Within, main run": fmt(main["within (pp)"]),
                         "Within, 5 seeds": rng(s["within min"], s["within max"]),
                         "Lowest ARI vs main run": f"{s['lowest ARI vs main']:.2f}",
                         "20 dimensions, mix / within": d20})
    return pd.DataFrame(rows)


def make_figure(beh, mx, dec, path):
    """A: first-20 popular share of every group (k = 4), late v3 to early v4. B: mix vs within, every grouping."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.2), gridspec_kw={"width_ratios": [1, 1.7]})
    for x0, t, color, name in [(0, beh, "#73726c", "Behavior groups"), (2.4, mx, "#104281", "Matrix groups")]:
        for _, g in t.iterrows():
            a.plot([x0, x0 + 1], [g["first20_popular_late_v3"], g["first20_popular_early_v4"]],
                   color=color, marker="o", ms=3, lw=1.5)
        a.text(x0 + 0.5, 74, name, ha="center", fontsize=7.5, color=color)
    a.set_xticks([0, 1, 2.4, 3.4], ["late v3", "early v4"] * 2, fontsize=6.5)
    a.set_ylim(0, 80)
    a.set_xlim(-0.4, 3.8)
    a.set_ylabel("First-20 popular share (%)", fontsize=8)
    a.set_title("A. Every group rises (k = 4)", loc="left", fontsize=8.5)

    labels = ["Rule-based\n(12)"] + [f"Behavior\nk = {k}" for k in p.KS] + [f"Matrix\nk = {k}" for k in p.KS]
    x = np.arange(len(dec))
    for offset, col, color, name in [(-0.2, "composition (pp)", "#eb6834", "Mix of groups"),
                                     (0.2, "within (pp)", "#1baf7a", "Within groups")]:
        v = dec[col].astype(float).to_numpy()
        b.bar(x + offset, v, width=0.38, color=color, label=name)
        for xi, vi in zip(x + offset, v):
            b.text(xi, max(vi, 0) + 1, f"{round(vi, 1) + 0.0:+.1f}", ha="center", fontsize=5.5)
    b.axhline(0, color="#30302e", lw=0.8)
    b.set_xticks(x, labels, fontsize=5.5)
    b.set_ylim(-5, 55)
    b.set_ylabel("Change, late v3 to early v4 (pp)", fontsize=8)
    b.set_title("B. Change in first-20 popular share, split by grouping", loc="left", fontsize=8.5)
    b.legend(frameon=False, fontsize=7, ncol=2, loc="upper center")
    for ax in (a, b):
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    # Sort by userId: the cached table has no fixed row order, and k-means and the SVD depend on it.
    u = p.load_users().sort_values("userId").reset_index(drop=True)  # same users as CP2 Finding 2

    # Behavior groups and the CP2 decomposition, reusing checkpoint2_profiles (they print their own tables)
    with contextlib.redirect_stdout(io.StringIO()):
        u = p.cluster(u)                   # adds the columns k3, k4, k5, k12
        behavior_dec = p.decomposition(u)  # rule-based row and behavior k-means rows

    # Matrix groups: 50 dimensions (main) and 20 dimensions (check)
    X = movie_matrix(u)
    labels, Z, variance = matrix_groups(X, MAIN_DIMS)
    labels20, _, variance20 = matrix_groups(X, CHECK_DIMS)
    print(f"SVD keeps {100 * variance:.1f}% of the variance with {MAIN_DIMS} dimensions, "
          f"{100 * variance20:.1f}% with {CHECK_DIMS}")
    for k in p.KS:
        u[f"mx{k}"] = labels[k]            # matrix groups: columns mx3, mx4, mx5, mx12

    low, high = u["later_pop"].quantile([1 / 3, 2 / 3])
    behavior = {k: group_table(u, f"k{k}", low, high) for k in p.KS}
    matrix = {k: group_table(u, f"mx{k}", low, high) for k in p.KS}
    matrix_dec = pd.DataFrame({f"matrix, k = {k}": decompose_labels(u, labels[k]) for k in p.KS}).T
    matrix_dec20 = pd.DataFrame({f"matrix, k = {k}": decompose_labels(u, labels20[k]) for k in p.KS}).T
    dec = pd.concat([behavior_dec, matrix_dec])

    rule = u["engagement"] + " | " + u["taste"]  # the 12 rule-based types of Checkpoint 1
    ari = pd.DataFrame({
        "matrix vs behavior": [adjusted_rand_score(u[f"mx{k}"], u[f"k{k}"]) for k in p.KS],
        "matrix vs rule-based": [adjusted_rand_score(u[f"mx{k}"], rule) for k in p.KS],
        "behavior vs rule-based": [adjusted_rand_score(u[f"k{k}"], rule) for k in p.KS],
    }, index=[f"k = {k}" for k in p.KS])

    print("\n=== Decomposition of the late v3 -> early v4 change in first-20 popular share (pp) ===")
    print(dec[["empty groups", "total (pp)", "composition (pp)", "within (pp)"]].astype(float).round(2).to_string())
    print(f"\n=== Same for matrix groups with {CHECK_DIMS} dimensions ===")
    print(matrix_dec20[["empty groups", "composition (pp)", "within (pp)"]].astype(float).round(2).to_string())

    # Seed check: slow, because every matrix k-means is repeated 4 times
    print("\nSeed check (4 more seeds for every k-means) ...")
    seeds = seed_check(u, StandardScaler().fit_transform(u[p.FEATURES].to_numpy(dtype="float64")), Z, dec)

    # The tables of the report, already rounded
    tables = {"checkpoint3_table1.csv": report_table(behavior[4]),
              "checkpoint3_table_a1.csv": report_table(matrix[4]),
              "checkpoint3_table_a2.csv": ari_table(ari),
              "checkpoint3_table_a3.csv": stress_table(dec, seeds, matrix_dec20)}
    for name, t in tables.items():
        t.to_csv(ROOT / "reports" / name, index=False)
    print("\n=== Table A3 (stress tests), as written to the report ===")
    print(tables["checkpoint3_table_a3.csv"].to_string(index=False))

    records = lambda tables_: pd.concat(tables_, names=["k", "group"]).reset_index().to_dict("records")
    frame = lambda t: t.rename_axis("grouping").reset_index().to_dict("records")
    out = {"dimensions": {"main": MAIN_DIMS, "check": CHECK_DIMS},
           "variance_kept": {"main": variance, "check": variance20},
           "behavior_groups": records(behavior), "matrix_groups": records(matrix),
           "decomposition": frame(dec), "decomposition_20_dimensions": frame(matrix_dec20),
           "ari": frame(ari.rename_axis("k")), "seed_check": frame(seeds)}
    (ROOT / "reports/checkpoint3_groupings.json").write_text(json.dumps(out, indent=2, default=float))
    make_figure(behavior[4], matrix[4], dec, ROOT / "figures/cp3_groupings.png")
    print("\nAll internal checks passed. Written: reports/checkpoint3_table1.csv, _a1, _a2, _a3, "
          "reports/checkpoint3_groupings.json, figures/cp3_groupings.png")


if __name__ == "__main__":
    main()