"""Checkpoint 2, profile stress test: does Finding 3's conclusion survive a grouping we did not choose?

Finding 3 (Checkpoint 1) grouped users with hand-picked rules (engagement x taste). Here we group the same users
with k-means instead and repeat the composition-vs-within decomposition of the late v3 -> early v4 change.

Run from the repo root (or from src/):
    python src/checkpoint2_profiles.py
Steps:
    1. load_users():      Finding 3's user table (checkpoint1.user_table) + 5 clustering features, cached as parquet
    2. cluster():         k-means on the standardized features for several k, with an independent sanity check
    3. agreement():       crosstabs and Adjusted Rand Index against the rule-based profiles
    4. decomposition():   Finding 3's composition-vs-within decomposition, with k-means groups instead of rule-based types
Writes:
    data/processed/profile_users.parquet     users + features (cache, not committed to git)
    figures/cp2_profile_decomposition.png    who-joined vs within-group change, by grouping
"""
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from checkpoint1 import (  # same profiles, launch date and decomposition as Finding 3
    V4,
    decompose,
    user_table,
)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
POP_DB = ROOT / "data/processed/eda.duckdb"  # built by src/checkpoint1.py
USERS_CACHE = ROOT / "data/processed/profile_users.parquet"
NO_TASTE = "too few later ratings (<40 total)"  # label used by checkpoint1.py for users without a taste type

# Five features, each tied to one idea. The first-20 popular share is left out on purpose:
# it is the outcome of Finding 3 and mostly reflects the sign-up screen (Finding 2), so using it would be circular.
FEATURES = ["log_n", "log_tenure", "log_first_session", "later_pop", "mean_rating"]
KS = [3, 4, 5, 12]  # 12 matches the number of rule-based types (4 engagement x 3 taste)
WINDOW_START, WINDOW_END = "2012-11-01", "2016-11-01"  # late v3 vs early v4; same windows as checkpoint1.decompose()
OUTCOME = "f20_t100t"  # first-20 share on top-100 movies of the previous 12 months


# ---------------------------------------------------------------------------
# Users and features
# ---------------------------------------------------------------------------
def load_users():
    """Users who have a rule-based taste type, with the 5 features. Cached because user_table takes a minute.

    Delete data/processed/profile_users.parquet if the feature definitions below change.
    """
    if USERS_CACHE.exists():
        return pd.read_parquet(USERS_CACHE)
    con = duckdb.connect(str(POP_DB), read_only=True)
    u, _ = user_table(con)  # scans all 32M ratings
    con.close()

    u = u[u["taste"] != NO_TASTE].copy()
    u["log_n"] = np.log1p(u["n"])                            # volume: how much the user rated
    u["log_tenure"] = np.log1p(u["tenure_days"])             # engagement: how long they stayed
    u["log_first_session"] = np.log1p(u["first_session_n"])  # size of the first sitting (batch import or not)
    u["later_pop"] = u["later_main"]                         # taste: popular share after the first 20 ratings
    u.to_parquet(USERS_CACHE)
    return u


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------
def check_assignment(X, centers, labels, n=5000, seed=0):
    """Independent check of sklearn's labels: recompute the nearest center by plain broadcasting (no matmul).

    sklearn printed matmul RuntimeWarnings on this machine; this confirms the labels are still right.
    Returns the share of sampled users whose nearest center equals their label (should be 1.0).
    """
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    dist = ((X[idx][:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return float((dist.argmin(axis=1) == labels[idx]).mean())


def cluster(u):
    """Standardize the features, run k-means for each k, print cluster profiles, and add labels k3, k4, k5, k12 to u."""
    X = StandardScaler().fit_transform(u[FEATURES].to_numpy(dtype="float64"))  # each feature: mean 0, sd 1
    assert np.isfinite(X).all(), "non-finite values in the standardized features"

    for k in KS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # spurious matmul warnings; verified below
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
            # silhouette is slow on 125k users, so score a random sample of 10,000
            sil = silhouette_score(X, km.labels_, sample_size=10_000, random_state=0)
        agree = check_assignment(X, km.cluster_centers_, km.labels_)
        u[f"k{k}"] = km.labels_

        print(f"\n=== k = {k}  (silhouette on a 10,000-user sample: {sil:.2f}; independent check agrees: {agree:.1%}) ===")
        summary = u.groupby(f"k{k}")[FEATURES].mean().round(2)  # cluster profile in the original (log) units
        summary.insert(0, "users", u.groupby(f"k{k}").size())
        print(summary)
    return u


# ---------------------------------------------------------------------------
# Agreement between k-means clusters and the rule-based profiles
# ---------------------------------------------------------------------------
def agreement(u, k):
    """Crosstabs of the k-means clusters against the rule-based types, plus Adjusted Rand Index (ARI).

    ARI: 1 = the two groupings are identical, 0 = no better than random, negative = worse than random.
    """
    col = f"k{k}"
    u = u.assign(rule_combined=u["engagement"] + " | " + u["taste"])  # the 4 x 3 cells used in Finding 3

    print(f"\n=== Agreement, k = {k} ===")
    for rule in ["engagement", "taste"]:
        # Row percentages: of the users in each cluster, what share falls in each rule-based type
        ct = pd.crosstab(u[col], u[rule], normalize="index").mul(100).round(1)
        print(f"\n{col} (rows) vs {rule} (columns), % of each cluster:")
        print(ct)

    print("\nAdjusted Rand Index against each rule-based grouping:")
    for rule in ["engagement", "taste", "rule_combined"]:
        print(f"  {rule:>14}: {adjusted_rand_score(u[rule], u[col]):.3f}")


# ---------------------------------------------------------------------------
# Does the Finding 3 decomposition hold when the profiles come from k-means?
# ---------------------------------------------------------------------------
def decompose_cells(u, cells, outcome=OUTCOME):
    """Same shift-share decomposition as checkpoint1.decompose(), but the grouping columns are an argument.

    total = composition (different mix of groups) + within (same groups behaving differently).
    Values are returned in percentage points.
    """
    a = u[(u["start"] >= WINDOW_START) & (u["start"] < V4)]   # late v3
    b = u[(u["start"] >= V4) & (u["start"] < WINDOW_END)]     # early v4
    t = pd.DataFrame({
        "share_a": a.groupby(cells).size() / len(a),
        "share_b": b.groupby(cells).size() / len(b),
        "y_a": a.groupby(cells)[outcome].mean(),
        "y_b": b.groupby(cells)[outcome].mean(),
    })
    n_empty = int(t.isna().any(axis=1).sum())  # groups that exist in only one of the two periods
    t = t.fillna(0)
    composition = ((t["share_b"] - t["share_a"]) * (t["y_a"] + t["y_b"]) / 2).sum()
    within = ((t["y_b"] - t["y_a"]) * (t["share_a"] + t["share_b"]) / 2).sum()
    total = b[outcome].mean() - a[outcome].mean()
    assert abs(total - composition - within) < 1e-9  # the two parts must add up to the total
    return {"users late v3": len(a), "users early v4": len(b), "groups": len(t), "empty groups": n_empty,
            "total (pp)": 100 * total, "composition (pp)": 100 * composition, "within (pp)": 100 * within}


def decomposition(u):
    """Rule-based baseline on the same users, then k-means groupings. Values in percentage points (pp)."""
    ours = decompose_cells(u, ["engagement", "taste"])

    # Consistency check: on the same grouping, our function must reproduce checkpoint1.decompose() exactly.
    base = decompose(u, OUTCOME)
    assert abs(100 * base["composition"] - ours["composition (pp)"]) < 1e-9
    assert abs(100 * base["within"] - ours["within (pp)"]) < 1e-9

    rows = {"rule-based (engagement x taste)": ours}
    for k in KS:
        rows[f"k-means, k = {k}"] = decompose_cells(u, [f"k{k}"])
    print("\n=== Decomposition of the late v3 -> early v4 change in first-20 popular share ===")
    print("(consistency check against checkpoint1.decompose passed)")
    table = pd.DataFrame(rows).T
    print(table.round(2))
    return table


def make_decomposition_figure(table, path):
    """Bars of 'who joined' (composition) and within-group change for each way of grouping users.

    Same colors and labels as Figure 4 of Checkpoint 1: orange = who joined, green = within types.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Rule-based\n(12 types)"] + [f"k-means\nk = {k}" for k in KS]  # same order as the rows of `table`
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(5.6, 3))
    for offset, col, color, name in [(-0.2, "composition (pp)", "#eb6834", "Who joined"),
                                     (0.2, "within (pp)", "#1baf7a", "Within types")]:
        ax.bar(x + offset, table[col], width=0.38, color=color, label=name)
        for xi, v in zip(x + offset, table[col]):
            label = "0.0" if abs(v) < 0.05 else f"{v:+.1f}"  # avoid printing "-0.0"
            ax.text(xi, max(v, 0) + 1, label, ha="center", fontsize=7)
    ax.axhline(0, color="#30302e", lw=0.8)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylim(-5, 55)
    ax.set_ylabel("Change, late v3 to early v4\n(percentage points)")
    # Legend above the plot, as in Figure 4 of Checkpoint 1
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)  # show every column of the decomposition table
    (ROOT / "figures").mkdir(exist_ok=True)

    u = load_users()
    print(f"Users with a rule-based type: {len(u):,}")

    u = cluster(u)
    for k in KS:
        agreement(u, k=k)
    table = decomposition(u)
    make_decomposition_figure(table, ROOT / "figures/cp2_profile_decomposition.png")


if __name__ == "__main__":
    main()