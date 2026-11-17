"""Checkpoint 3, Finding 3: are a user's first 20 ratings closer to their own later ratings than to other users'?

H0 (per user): a user's first 20 ratings are no more similar to their own ratings 21+ than to the ratings 21+ of a
random user from the same site version who joined in the same calendar month (same catalog, same popular movies)
and is in the same rating-volume band. A stronger null also matches donors on first-20 popular share, so any
self-similarity left cannot come from how popular the early movies were.

Test: for every user, compare their early profile with their own later profile and with the later profiles of
K donors drawn from their stratum. Self-rank = share of donors less similar than the user's own later ratings
(ties count half). Under H0 self-rank is uniform with mean 0.5; its mean is the probability that a user's early
ratings are closer to their own later ratings than to a same-stratum stranger's.

No-repeat correction: nobody rates a movie twice, so a user's own later ratings never contain their first 20 movies,
while a donor's later ratings can. Without a correction this tilts the test against the user (v4 users rate the most
popular movies first, and donors rate those same movies later). Likewise a donor's later ratings never contain the
donor's own first 20. So every user-donor comparison drops both first-20 sets from both later profiles: the user's
later ratings minus the donor's first 20 versus the donor's later ratings minus the user's first 20.

Four ways to measure closeness (early = ratings 1-20, later = ratings 21+):
    genre       cosine between genre-label counts
    popularity  minus |difference| in share of ratings on the previous-12-month top 100 (checkpoint1's top100_trail)
    stars       minus |difference| in mean star rating
    movie space cosine in a 50-dimension SVD space fit on the user x movie matrix of ratings 21+ (as in Finding 2);
                early and later sets never share a movie, so closeness here means "rated movies that the same
                people rate", not "rated the same movies"
Runs: v4+ (main, one site version as the instructor suggested) and v3 (replication); minimum ratings n = 100, 60, 40
(n >= 40 is the lowest that leaves 20 later ratings); later = all ratings 21+ or only those entered more than a day
after the first rating (about half of users enter everything in one sitting).
Calibration check (assert): replacing every user's own later profile with a random donor's (with the same exclusions)
must give mean self-rank 0.5.
Hypothesis test: z = (mean self-rank - 0.5) / standard error, two-sided p from the normal distribution.

Run after src/checkpoint1.py has built data/processed/eda.duckdb:
    python src/checkpoint3_self_similarity.py
Writes:
    reports/checkpoint3_self_similarity.csv   every run: mean self-rank (95% CI), share in top 5%, users
    reports/checkpoint3_self_similarity.json  the same plus the histograms behind the figure
    figures/cp3_self_similarity.png           Figure 3
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import norm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import checkpoint2_taste as taste  # run_sql, V3, V4, MOVIES: same definitions as Findings 1-2

K = 50              # donors per user
MIN_STRATUM = 11    # a user needs at least 10 possible donors in their stratum
EARLY = 20
THRESHOLDS = (100, 60, 40)
DIMS = 50           # same as Finding 2's matrix groups
MEASURES = ["genre", "popularity", "stars", "movie space"]
NULLS = {"month + volume": False, "+ first-20 popularity": True}
LATER = ["all", "after day 1"]
DAY = 86400
CHUNK = 2000
BINS = (np.arange(0, 101, 10) - 0.5) / 100  # 10 bins; edges sit between the 0.01 steps a rank can take
BINS[-1] = 1.005
warnings.simplefilter("ignore", RuntimeWarning)  # spurious matmul warnings on this machine (as in Finding 2); asserted below


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load():
    """Users (sorted by userId) and their ratings, as numpy arrays.
    Returns u (one row per user), E (N x 20 early movieIds), later tables per variant, and movie-level vectors."""
    users_sql = f"""
with u as (
    select userId, count(*) as n, min(ts) as first_ts from seqp group by userId
    having count(*) >= {min(THRESHOLDS)} and min(ts) >= epoch(timestamp '{taste.V3}')
)"""
    u = taste.run_sql(users_sql + " select * from u order by userId").set_index("userId")
    r = taste.run_sql(users_sql + f"""
select s.userId, s.movieId, s.ord <= {EARLY} as early, s.ts >= u.first_ts + {DAY} as d1,
       s.top100_trail::tinyint as pop, s.rating::float as stars
from seqp s join u using (userId) order by s.userId, s.ord""")
    print(f"{len(u):,} users (v3 + v4+, >= {min(THRESHOLDS)} ratings), {len(r):,} ratings")

    start = pd.to_datetime(u["first_ts"], unit="s")
    u["era"] = np.where(start < taste.V4, "v3", "v4+")
    u["month"] = start.dt.year * 12 + start.dt.month
    u["band"] = pd.cut(u["n"], [0, 99, 299, np.inf], labels=False).astype(int)  # 40-99, 100-299, 300+

    ui = u.index.get_indexer(r["userId"])
    mid, pop, stars = r["movieId"].to_numpy(), r["pop"].to_numpy(float), r["stars"].to_numpy(float)
    early = r["early"].to_numpy()
    m = pd.read_parquet(taste.MOVIES, columns=["movieId", "genres"])
    N, ncol = len(u), int(max(mid.max(), m["movieId"].max())) + 1
    E = mid[early].reshape(N, EARLY)  # rows are sorted by user then ord, and every user has exactly 20 early ratings
    assert (ui[early].reshape(N, EARLY) == np.arange(N)[:, None]).all()
    u["early_pop"] = pop[early].reshape(N, EARLY).mean(1)
    u["early_stars"] = stars[early].reshape(N, EARLY).mean(1)
    u["pop_bin"] = np.minimum((u["early_pop"] * 5).astype(int), 4)  # fifths of first-20 popular share

    later = {}
    for name, mask in [("all", ~early), ("after day 1", ~early & r["d1"].to_numpy())]:
        lu, lm = ui[mask], mid[mask]
        key = lu.astype(np.int64) * ncol + lm  # unique: a user rates a movie once
        o = np.argsort(key, kind="stable")
        later[name] = {"u": lu[o], "m": lm[o], "key": key[o], "popularity": pop[mask][o], "stars": stars[mask][o]}
        u[f"n_later|{name}"] = np.bincount(lu, minlength=N)

    g = m.assign(genre=m["genres"].str.split("|")).explode("genre")
    g = g[~g["genre"].isin(["(no genres listed)", "IMAX"])]
    gcols = pd.Index(sorted(g["genre"].unique()))
    gm = csr_matrix((np.ones(len(g)), (g["movieId"].to_numpy(), gcols.get_indexer(g["genre"]))),
                    shape=(ncol, len(gcols))).toarray()  # movie x genre, 0/1
    B = csr_matrix((np.ones(len(later["all"]["u"])), (later["all"]["u"], later["all"]["m"])), shape=(N, ncol))
    svd = TruncatedSVD(n_components=DIMS, random_state=0).fit(normalize(B))  # same fit as Finding 2, ratings 21+
    vm = svd.components_.T.copy()  # movie x 50; a set of movies sits at the sum of its movies' rows
    assert np.isfinite(vm).all()
    print(f"SVD keeps {100 * svd.explained_variance_ratio_.sum():.1f}% of the variance with {DIMS} dimensions")
    return u, E, later, {"genre": gm, "movie space": vm}, ncol


def profiles(u, E, later, vecs, ncol):
    """Early profiles (per user) and raw later sums (per user and variant), so overlaps can be subtracted later."""
    early = {"popularity": u["early_pop"].to_numpy(), "stars": u["early_stars"].to_numpy()}
    Be = csr_matrix((np.ones(E.size), (np.repeat(np.arange(len(u)), EARLY), E.ravel())), shape=(len(u), ncol))
    for m, v in vecs.items():
        early[m] = unit(Be @ v)
    sums = {}
    for name, t in later.items():
        B = csr_matrix((np.ones(len(t["u"])), (t["u"], t["m"])), shape=(len(u), ncol))
        sums[name] = {"n": u[f"n_later|{name}"].to_numpy(float),
                      "popularity": np.bincount(t["u"], t["popularity"], len(u)),
                      "stars": np.bincount(t["u"], t["stars"], len(u)),
                      **{m: B @ v for m, v in vecs.items()}}
    return early, sums


def unit(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x, dtype=float), where=norm > 0)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------
def sample_donors(keys, rng):
    """K donors per user from users with the same key, never the user themself (with replacement).
    Returns donor indices (N x K) and a mask of users whose stratum is big enough."""
    order = np.argsort(keys, kind="stable")
    _, start, size = np.unique(keys[order], return_index=True, return_counts=True)
    grp = np.repeat(np.arange(len(start)), size)                 # group of each sorted position
    pos = np.arange(len(keys)) - start[grp]                       # position inside its group
    r = (rng.random((len(keys), K)) * (size[grp] - 1)[:, None]).astype(int)
    r += r >= pos[:, None]                                        # skip the user's own position
    r = np.minimum(r, (size[grp] - 1)[:, None])                   # one-user strata stay in bounds (masked out below)
    donors_sorted = order[start[grp][:, None] + r]
    donors, ok = np.empty_like(donors_sorted), np.empty(len(keys), bool)
    donors[order], ok[order] = donors_sorted, size[grp] >= MIN_STRATUM
    return donors, ok


def similarity(early, sums, t, vecs, E, ncol, idx, d, drop):
    """Similarity of user idx[i]'s early profile with the later profile of d[i, k], for every measure (n x K each).
    The later profile of d[i, k] leaves out the first-20 movies of every user in drop[*][i, k] (no-repeat correction);
    each element of drop is an n x K array of user indices."""
    out = {m: np.empty(d.shape) for m in MEASURES}
    for c in range(0, len(idx), CHUNK):
        i, dd = idx[c:c + CHUNK], d[c:c + CHUNK]
        n, k = dd.shape
        pairs = []
        for x in drop:
            q = dd[:, :, None].astype(np.int64) * ncol + E[x[c:c + CHUNK]]     # (later owner, excluded movie)
            pos = np.minimum(np.searchsorted(t["key"], q), len(t["key"]) - 1)
            hit = t["key"][pos] == q                                           # the later owner rated it later
            a, b, _ = np.nonzero(hit)
            pairs.append((a * k + b).astype(np.int64) * len(t["key"]) + pos[hit])
        uniq = np.unique(np.concatenate(pairs))                                # a movie excluded twice counts once
        pair, ri = uniq // len(t["key"]), uniq % len(t["key"])
        over = lambda w: np.bincount(pair, w, n * k).reshape(n, k)
        cnt = sums["n"][dd] - over(None)
        assert (cnt > 0).all()
        for m in ["popularity", "stars"]:
            out[m][c:c + CHUNK] = -np.abs(early[m][i][:, None] - (sums[m][dd] - over(t[m][ri])) / cnt)
        P = csr_matrix((np.ones(len(pair)), (pair, t["m"][ri])), shape=(n * k, ncol))  # overlapping movies per pair
        for m, v in vecs.items():
            lat = unit(sums[m][dd] - (P @ v).reshape(n, k, -1))
            out[m][c:c + CHUNK] = np.einsum("nd,nkd->nk", early[m][i], lat)
    return out


def self_rank(self_sim, donor_sim):
    """Share of donors less similar than the user's own later ratings (ties count half). Both n x K: comparison k
    pairs the user's own profile and donor k's profile, each with the same movies dropped."""
    return ((donor_sim < self_sim).sum(1) + 0.5 * (donor_sim == self_sim).sum(1)) / donor_sim.shape[1]


def run(data, era, n_min, later_name, rng):
    """Self-rank of every eligible user, for each null and measure. Returns (summary rows, rank arrays)."""
    u, E, later, vecs, ncol, early, sums = data
    sim = lambda idx, d, drop: similarity(early, sums[later_name], later[later_name], vecs, E, ncol, idx, d, drop)
    keep = ((u["era"] == era) & (u["n"] >= n_min) & (u[f"n_later|{later_name}"] >= EARLY)).to_numpy()
    sub = np.flatnonzero(keep)
    rows, ranks = [], {}
    for null, by_pop in NULLS.items():
        key = (u["month"].to_numpy() * 10 + u["band"].to_numpy()) * 10 + (u["pop_bin"].to_numpy() if by_pop else 0)
        donors, ok = sample_donors(key[sub], rng)
        idx, donors = sub[ok], sub[donors[ok]]
        me = np.repeat(idx[:, None], K, axis=1)
        own, others = sim(idx, me, [donors]), sim(idx, donors, [me])  # own minus donor's first 20; donor minus user's
        # Calibration: a random donor f in place of the user, with the same symmetric exclusions, must sit at 0.5
        f = np.repeat(donors[np.arange(len(idx)), rng.integers(0, K, len(idx))][:, None], K, axis=1)
        fake, others_f = sim(idx, f, [me, donors]), sim(idx, donors, [me, f])
        for m in MEASURES:
            r = self_rank(own[m], others[m])
            pr = self_rank(fake[m], others_f[m])
            placebo = pr.mean()
            assert abs(placebo - 0.5) < 0.02, (era, n_min, later_name, null, m, placebo)
            rows.append({"era": era, "n_min": n_min, "later": later_name, "null": null, "measure": m,
                         "users": len(idx), "dropped_small_strata": int((~ok).sum()),
                         "mean_self_rank": r.mean(), "ci95": 1.96 * r.std() / np.sqrt(len(r)),
                         "z": (r.mean() - 0.5) / (r.std() / np.sqrt(len(r))),
                         "p": 2 * norm.sf(abs(r.mean() - 0.5) / (r.std() / np.sqrt(len(r)))),
                         "share_top5": (r >= 0.95).mean(), "share_below_half": (r < 0.5).mean(),
                         "placebo_mean": placebo, "placebo_share_top5": (pr >= 0.95).mean()})
            ranks[(null, m)], ranks[(null, m, "placebo")] = r, pr
    return rows, ranks


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------
def make_figure(ranks, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"month + volume": "#2a78d6", "+ first-20 popularity": "#eb6834"}  # CP1 palette (validated)
    labels = {"month + volume": "Same month and volume", "+ first-20 popularity": "Also same first-20 popularity"}
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.6), sharey=True)
    for ax, m in zip(axes, MEASURES):
        for null in NULLS:
            r = ranks[(null, m)]
            h, _ = np.histogram(r, bins=BINS)
            ax.stairs(100 * h / len(r), BINS, color=colors[null], lw=2, label=labels[null])
            ax.axvline(r.mean(), color=colors[null], lw=1, ls=":")
        h, _ = np.histogram(ranks[("month + volume", m, "placebo")], bins=BINS)  # what chance alone gives
        ax.stairs(100 * h / h.sum(), BINS, color="#73726c", lw=1, ls="--")
        ax.set_title(m.capitalize(), loc="left", fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Self-rank", fontsize=7.5)
    axes[0].set_ylabel("Users (%)", fontsize=8)
    h, _ = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"Donors: {labels[n].lower()}" for n in NULLS] , frameon=False, fontsize=7, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.text(0.5, -0.09, "Dotted: mean self-rank. Gray dashed: chance (a random donor in place of the user).",
             ha="center", fontsize=6.5, color="#73726c")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    pd.set_option("display.width", 250)
    u, E, later, vecs, ncol = load()
    data = (u, E, later, vecs, ncol, *profiles(u, E, later, vecs, ncol))
    rng = np.random.default_rng(0)

    rows, fig_ranks, hists = [], None, {}
    for era in ["v4+", "v3"]:
        for n_min in THRESHOLDS:
            for later_name in LATER:
                if later_name == "after day 1" and n_min != 100:
                    continue  # the day-1 check is run at the main threshold only
                r, ranks = run(data, era, n_min, later_name, rng)
                rows += r
                hists[f"{era} | n>={n_min} | {later_name}"] = {
                    f"{null} | {m}": np.histogram(v, bins=BINS)[0].tolist()
                    for (null, m, *_), v in ranks.items() if not _}
                if (era, n_min, later_name) == ("v4+", 100, "all"):
                    fig_ranks = ranks
                print(f"  done: {era}, n >= {n_min}, later = {later_name}")

    res = pd.DataFrame(rows)
    show = res.assign(mean=res["mean_self_rank"].round(3), ci=res["ci95"].round(3),
                      top5=(100 * res["share_top5"]).round(1), below_half=(100 * res["share_below_half"]).round(1))
    print(show[["era", "n_min", "later", "null", "measure", "users", "mean", "ci", "z", "p", "top5", "below_half",
                "placebo_mean"]].to_string(index=False))

    res.to_csv(ROOT / "reports/checkpoint3_self_similarity.csv", index=False)
    (ROOT / "reports/checkpoint3_self_similarity.json").write_text(json.dumps(
        {"K": K, "min_stratum": MIN_STRATUM, "dims": DIMS, "results": res.to_dict("records"),
         "histograms_10_bins": hists}, indent=2, default=float))
    make_figure(fig_ranks, ROOT / "figures/cp3_self_similarity.png")
    print("\nCalibration checks passed. Written: reports/checkpoint3_self_similarity.csv/.json, "
          "figures/cp3_self_similarity.png")


if __name__ == "__main__":
    main()
