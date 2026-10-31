"""Checkpoint 2, Finding B: genre taste over the user lifecycle, before vs after the v4 launch.

Run after src/checkpoint1.py has built data/processed/eda.duckdb (table `seqp`):
    python src/checkpoint2_taste.py
Writes:
    figures/cp2_genre_lifecycle.png   all eras, 3 rating-volume bands (same layout as Figure 5 of Checkpoint 1)
    figures/cp2_genre_cohorts.png     stress test: users who joined 2 years before vs after v4
    reports/checkpoint2_metrics.json
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POP_DB = ROOT / "data/processed/eda.duckdb"  # built by src/checkpoint1.py
MOVIES = ROOT / "data/processed/movies.parquet"
V3, V4 = "2003-02-18", "2014-11-01"
COHORT_START, COHORT_END = "2012-11-01", "2016-11-01"  # 2 years before / after the v4 launch (as in Finding 3)
COHORTS = ["late v3", "early v4"]
BANDS = ["100-299", "300-999", "1000+"]
TENTHS = list(range(1, 11))
ERA_COLORS = {"pre-v3": "#2a78d6", "v3": "#eb6834", "v4+": "#1baf7a"}  # same colors as checkpoint1.py

# Both analyses use the same users as checkpoint1.py's lifecycle(): >= 100 ratings, first rating in 1998 or later.
USERS_SQL = """
with u as (
    select userId, count(*) as n, min(ts) as first_ts
    from seqp
    group by userId
    having count(*) >= 100 and min(ts) >= epoch(timestamp '1998-01-01')
)
"""


def run_sql(q):
    con = duckdb.connect(str(POP_DB), read_only=True)
    df = con.execute(q).fetchdf()
    con.close()
    return df


# ---------------------------------------------------------------------------
# A. All eras x three rating-volume bands (comparable to Figure 5 of Checkpoint 1)
# ---------------------------------------------------------------------------
def load_lifecycle_stats():
    """Mean share (%) of each genre in a user's own ratings, by era x volume band x tenth, with 95% CI.

    Computed in SQL so only a small table comes back: for each group we keep the sum and the sum of squares
    of the per-user shares (a user who never rated a genre in a tenth counts as 0).
    """
    q = USERS_SQL + f""",
ug as (
    select *,
        case when first_ts < epoch(timestamp '{V3}') then 'pre-v3'
             when first_ts < epoch(timestamp '{V4}') then 'v3'
             else 'v4+' end as era,
        case when n < 300 then '100-299' when n < 1000 then '300-999' else '1000+' end as band
    from u
),
picked as (
    select s.userId, s.movieId, ug.era, ug.band,
           floor((s.ord - 1) * 10.0 / ug.n)::int + 1 as tenth
    from seqp s
    join ug using (userId)
),
tagged as (
    select p.userId, p.era, p.band, p.tenth, unnest(string_split(m.genres, '|')) as genre
    from picked p
    join '{MOVIES}' m using (movieId)
),
counts as (
    select userId, era, band, tenth, genre, count(*) as c
    from tagged
    where genre not in ('(no genres listed)', 'IMAX')
    group by all
),
shares as (
    select *, 100.0 * c / sum(c) over (partition by userId, tenth) as share
    from counts
),
group_n as (
    select era, band, count(*) as n_users from ug group by all
)
select s.era, s.band, s.tenth, s.genre, g.n_users,
       sum(s.share) as sum_share, sum(s.share * s.share) as sum_sq
from shares s
join group_n g using (era, band)
group by s.era, s.band, s.tenth, s.genre, g.n_users
"""
    df = run_sql(q)
    n = df["n_users"]
    df["mean"] = df["sum_share"] / n
    var = (df["sum_sq"] - n * df["mean"] ** 2) / (n - 1)
    df["ci95"] = 1.96 * np.sqrt(var.clip(lower=0) / n)  # same form as mean_ci() in checkpoint1.py
    return df[["era", "band", "tenth", "genre", "n_users", "mean", "ci95"]]


def pick_genres(stats, k=3, band="100-299"):
    """Genres whose tenth-1 share differs most between v3 and v4+ users (in one volume band)."""
    t = stats[(stats["band"] == band) & (stats["tenth"] == 1)].pivot(index="genre", columns="era", values="mean")
    return list((t["v4+"] - t["v3"]).abs().sort_values(ascending=False).index[:k])


def make_lifecycle_figure(stats, genres, path):
    """Same layout as Figure 5 of Checkpoint 1: columns = volume bands, lines = eras, x = tenth. One row per genre."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(genres), 3, figsize=(7.2, 2.3 * len(genres)),
                             sharex=True, sharey="row", squeeze=False)
    for i, genre in enumerate(genres):
        for ax, band in zip(axes[i], BANDS):
            for era, color in ERA_COLORS.items():
                s = stats[(stats["genre"] == genre) & (stats["band"] == band) & (stats["era"] == era)].sort_values("tenth")
                ax.plot(s["tenth"], s["mean"], color=color, lw=2, marker="o", ms=3,
                        label=f"{era} (n={s['n_users'].iloc[0]:,})")
                ax.fill_between(s["tenth"], s["mean"] - s["ci95"], s["mean"] + s["ci95"], color=color, alpha=0.2, lw=0)
            if i == 0:
                ax.set_title(f"Users with {band} ratings", loc="left", fontsize=8.5)
                ax.legend(frameon=False, fontsize=6.5)
            ax.set_xticks(TENTHS)
            ax.tick_params(labelsize=7)
            ax.grid(color="#e6e6e3", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        axes[i, 0].set_ylabel(f"{genre}\nshare of ratings (%)", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Tenth of user's rating history", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# B. Stress test: users who joined 2 years before vs after v4 (same window as Finding 3)
# ---------------------------------------------------------------------------
def load_cohort_counts():
    """One row per (user, tenth, popular flag, genre): number of ratings in that genre.

    popular = 1 if the movie was in the top 100 of the previous 12 months (checkpoint1's top100_trail).
    """
    q = USERS_SQL + f""",
picked as (
    select s.userId, s.movieId, s.top100_trail as popular,
           floor((s.ord - 1) * 10.0 / u.n)::int + 1 as tenth,
           case when u.first_ts < epoch(timestamp '{V4}') then 'late v3' else 'early v4' end as cohort
    from seqp s
    join u using (userId)
    where u.first_ts >= epoch(timestamp '{COHORT_START}')
      and u.first_ts <  epoch(timestamp '{COHORT_END}')
),
tagged as (
    select p.userId, p.cohort, p.tenth, p.popular, unnest(string_split(m.genres, '|')) as genre
    from picked p
    join '{MOVIES}' m using (movieId)
)
select userId, cohort, tenth, popular, genre, count(*) as c
from tagged
where genre not in ('(no genres listed)', 'IMAX')
group by all
"""
    return run_sql(q)


def mean_shares(counts, popular=None):
    """Mean share (%) of each genre per user and tenth, averaged within cohort, with 95% CI.

    popular=None uses all ratings; popular=0 keeps only ratings NOT on top-100 movies.
    Users with no ratings left in a tenth are dropped from that tenth.
    Returns (mean, ci, n_users), each indexed by (cohort, tenth).
    """
    d = counts if popular is None else counts[counts["popular"] == popular]
    d = d.groupby(["userId", "cohort", "tenth", "genre"], as_index=False)["c"].sum()
    d["share"] = 100 * d["c"] / d.groupby(["userId", "tenth"])["c"].transform("sum")
    wide = d.pivot_table(index=["cohort", "tenth", "userId"], columns="genre", values="share", fill_value=0)
    g = wide.groupby(["cohort", "tenth"])
    n = g.size()
    mean = g.mean()
    ci = 1.96 * g.std().div(np.sqrt(n), axis=0)
    return mean, ci, n


def summary_table(mean, ci, genres, tenths=(1, 5, 10)):
    """Text table: 'mean ± CI' for each cohort x selected tenth, one row per genre."""
    out = {}
    for c in COHORTS:
        for t in tenths:
            out[f"{c}, tenth {t}"] = [f"{mean.loc[(c, t), k]:.1f} ± {ci.loc[(c, t), k]:.1f}" for k in genres]
    return pd.DataFrame(out, index=genres)


def make_cohort_figure(mean, ci, genres, path):
    """One small panel per genre: share across the ten tenths, one line per cohort (95% CI band)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"late v3": ERA_COLORS["v3"], "early v4": ERA_COLORS["v4+"]}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for ax, genre in zip(axes.ravel(), genres):
        for cohort, color in colors.items():
            ys = np.array([mean.loc[(cohort, t), genre] for t in TENTHS])
            es = np.array([ci.loc[(cohort, t), genre] for t in TENTHS])
            ax.plot(TENTHS, ys, color=color, marker="o", ms=3, lw=2, label=cohort)
            ax.fill_between(TENTHS, ys - es, ys + es, color=color, alpha=0.2, lw=0)
        ax.set_title(genre, loc="left", fontsize=9)
        ax.set_xticks(TENTHS)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.supxlabel("Tenth of the user's rating history", fontsize=8)
    fig.supylabel("Share of a user's ratings in this genre (%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    (ROOT / "figures").mkdir(exist_ok=True)

    # A. all eras x volume bands
    stats = load_lifecycle_stats()
    genres_a = pick_genres(stats, k=3)
    print("A. All eras. Genres shown (largest v3 vs v4+ gap in tenth 1, 100-299 band):", genres_a)
    show = stats[stats["genre"].isin(genres_a) & stats["tenth"].isin([1, 5, 10])]
    print(show.pivot_table(index=["genre", "band", "tenth"], columns="era", values="mean").round(1), "\n")
    make_lifecycle_figure(stats, genres_a, ROOT / "figures/cp2_genre_lifecycle.png")

    # B. stress test with narrow cohorts
    counts = load_cohort_counts()
    mean, ci, n = mean_shares(counts)
    start_diff = (mean.loc[("early v4", 1)] - mean.loc[("late v3", 1)]).abs()
    genres_b = list(start_diff.sort_values(ascending=False).index[:6])
    mean_u, ci_u, n_u = mean_shares(counts, popular=0)  # drop ratings on top-100 movies
    print("B. Users per cohort and tenth (all ratings):")
    print(n.unstack("tenth").to_string(), "\n")
    print("B. ALL RATINGS: mean share of a user's ratings (%) +- 95% CI")
    print(summary_table(mean, ci, genres_b), "\n")
    print("B. STRESS TEST, ratings on top-100 movies removed. Users per cohort and tenth:")
    print(n_u.unstack("tenth").to_string())
    print(summary_table(mean_u, ci_u, genres_b), "\n")
    make_cohort_figure(mean, ci, genres_b, ROOT / "figures/cp2_genre_cohorts.png")

    metrics = {
        "lifecycle_all_eras": {"genres_shown": genres_a, "stats": stats.round(4).to_dict("records")},
        "cohorts": {
            "windows": {"late v3": [COHORT_START, V4], "early v4": [V4, COHORT_END]},
            "genres_shown": genres_b,
            "users": n.reset_index(name="users").to_dict("records"),
            "all_ratings": {"mean": mean.reset_index().to_dict("records"), "ci95": ci.reset_index().to_dict("records")},
            "without_top100": {"users": n_u.reset_index(name="users").to_dict("records"),
                               "mean": mean_u.reset_index().to_dict("records"),
                               "ci95": ci_u.reset_index().to_dict("records")},
        },
    }
    (ROOT / "reports/checkpoint2_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()