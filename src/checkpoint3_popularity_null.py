"""Checkpoint 3, Finding 1: are the late-v3 vs early-v4 genre gaps explained by popularity alone?

H0 (popularity-only): given the calendar month and a movie's popularity bin when it was rated, a user's choice of
movie is unrelated to genre. Under H0 a user's genre share is what we get by replacing each of their ratings with a
random rating from the same month and popularity bin. We use the expected value of that replacement (no simulation
needed) and compare the observed cohort gaps with the gaps expected under H0.

Depletion check: repeat the comparison keeping only user-tenths that start with at least MIN_UNRATED of the month's
top-100 movies not yet rated (see checkpoint3_depletion.py; same users: >= 100 ratings, same cohort windows).

Run after src/checkpoint1.py has built data/processed/eda.duckdb:
    python src/checkpoint3_popularity_null.py
Writes:
    figures/cp3_popularity_null.png
    reports/checkpoint3_popularity_null.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import checkpoint2_taste as taste  # noqa: E402
import checkpoint3_depletion as depletion  # noqa: E402

MIN_RATINGS = 100  # must match depletion.MIN_RATINGS so both scripts use the same users
MIN_UNRATED = 50  # depletion check: keep user-tenths with at least this many top-100 movies not yet rated
GENRES = ["Comedy", "Adventure", "Action", "Sci-Fi", "Romance", "Crime"]  # the six genres of CP2 Figure 2
GENRES_SHOWN = ["Comedy", "Action", "Crime"]  # figure
UNRATED_CACHE = ROOT / "data/processed/cp3_unrated_top100.parquet"  # data/ is untracked
BIN_SQLS = {  # popularity bin of a movie in the month it was rated (trail_rank = rank by ratings in the previous 12 months)
    "coarse": """case when p.trail12 = 0 then 'no recent ratings'
                      when p.trail_rank <= 100 then 'top 100'
                      when p.trail_rank <= 1000 then 'rank 101-1000'
                      else 'rank > 1000' end""",
    "fine": """case when p.trail12 = 0 then 'no recent ratings'
                    when p.trail_rank <= 10 then 'top 10'
                    when p.trail_rank <= 30 then 'rank 11-30'
                    when p.trail_rank <= 100 then 'rank 31-100'
                    when p.trail_rank <= 300 then 'rank 101-300'
                    when p.trail_rank <= 1000 then 'rank 301-1000'
                    when p.trail_rank <= 3000 then 'rank 1001-3000'
                    else 'rank > 3000' end""",
}


def strata_path(spec):
    return ROOT / f"data/processed/cp3_strata_{spec}.parquet"  # data/ is untracked


def load_strata(spec):
    """Expected genre labels per rating for every (month, popularity bin): the H0 reference. Slow once, then cached."""
    if strata_path(spec).exists():
        return pd.read_parquet(strata_path(spec))
    q = f"""
with r as (
    select strftime(s.ym, '%Y-%m') as ym, {BIN_SQLS[spec]} as bin, s.movieId
    from seqp s join pop_rank p on p.movieId = s.movieId and p.ym = s.ym
),
n as (select ym, bin, count(*) as n_ratings from r group by all),
g as (
    select r.ym, r.bin, unnest(string_split(m.genres, '|')) as genre
    from r join '{taste.MOVIES}' m on m.movieId = r.movieId
)
select g.ym, g.bin, g.genre, count(*)::double / n.n_ratings as exp_per_rating
from g join n on n.ym = g.ym and n.bin = g.bin
where g.genre not in ('(no genres listed)', 'IMAX')
group by g.ym, g.bin, g.genre, n.n_ratings
"""
    df = taste.run_sql(q)
    df.to_parquet(strata_path(spec))
    return df


def load_user_bins(spec):
    """Number of each cohort user's ratings in every (tenth, month, popularity bin)."""
    q = taste.users_sql(MIN_RATINGS) + f"""
select s.userId,
       case when u.first_ts < epoch(timestamp '{taste.V4}') then 'late v3' else 'early v4' end as cohort,
       floor((s.ord - 1) * 10.0 / u.n)::int + 1 as tenth,
       strftime(s.ym, '%Y-%m') as ym, {BIN_SQLS[spec]} as bin, count(*) as k
from seqp s
join pop_rank p on p.movieId = s.movieId and p.ym = s.ym
join u on u.userId = s.userId
where u.first_ts >= epoch(timestamp '{taste.COHORT_START}') and u.first_ts < epoch(timestamp '{taste.COHORT_END}')
group by all
"""
    return taste.run_sql(q)


def null_counts(user_bins, strata):
    """Expected genre labels per (user, tenth) under H0: sum over the user's ratings of their stratum's expectation."""
    exp = strata.pivot_table(index=["ym", "bin"], columns="genre", values="exp_per_rating", fill_value=0)
    keys = pd.MultiIndex.from_frame(user_bins[["ym", "bin"]])
    e = pd.DataFrame(exp.reindex(keys).fillna(0).to_numpy() * user_bins["k"].to_numpy()[:, None], columns=exp.columns)
    for c in ["userId", "cohort", "tenth"]:
        e[c] = user_bins[c].to_numpy()
    return e.groupby(["userId", "cohort", "tenth"]).sum()


def observed_counts():
    c = taste.load_cohort_counts(MIN_RATINGS)
    return c.groupby(["userId", "cohort", "tenth", "genre"])["c"].sum().unstack("genre", fill_value=0)


def to_share(counts):
    return 100 * counts.div(counts.sum(axis=1), axis=0)


def summarize(obs_share, null_share):
    """Early-v4 minus late-v3 gap in genre share: observed, expected under H0, and the excess (observed - null)."""
    resid = obs_share - null_share  # per-user excess over the popularity-only expectation
    cohort = obs_share.index.get_level_values("cohort").to_numpy()
    tenth = obs_share.index.get_level_values("tenth").to_numpy()
    rows = []
    for g in GENRES:
        for t in range(1, 11):
            row = {"genre": g, "tenth": t}
            for name, df in [("observed", obs_share), ("null", null_share), ("excess", resid)]:
                a = df.loc[(cohort == "early v4") & (tenth == t), g]
                b = df.loc[(cohort == "late v3") & (tenth == t), g]
                row[f"{name}_gap"] = float(a.mean() - b.mean())
                row[f"{name}_ci95"] = float(1.96 * np.hypot(a.std() / np.sqrt(len(a)), b.std() / np.sqrt(len(b))))
            rows.append(row)
    return pd.DataFrame(rows)


def load_unrated_cached():
    """Top-100 movies not yet rated at the start of each (user, tenth); slow once (see checkpoint3_depletion.py)."""
    if UNRATED_CACHE.exists():
        return pd.read_parquet(UNRATED_CACHE)
    df = depletion.load_unrated()
    df.to_parquet(UNRATED_CACHE)
    return df


def not_depleted(index):
    """Boolean array over (userId, cohort, tenth) rows: True if the tenth starts with >= MIN_UNRATED unrated top-100."""
    u = load_unrated_cached()
    ok = pd.MultiIndex.from_frame(u.loc[u["unrated_top100"] >= MIN_UNRATED, ["userId", "tenth"]])
    return index.droplevel("cohort").isin(ok)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(GENRES_SHOWN), figsize=(7.4, 2.6), sharex=True)
    for ax, g in zip(axes, GENRES_SHOWN):
        d = res[res["genre"] == g]
        o = d[d["spec"] == "coarse"].sort_values("tenth")  # observed does not depend on the bin spec
        ax.plot(o["tenth"], o["observed_gap"], color="#104281", lw=2, marker="o", ms=3, label="Observed")
        ax.fill_between(o["tenth"], o["observed_gap"] - o["observed_ci95"], o["observed_gap"] + o["observed_ci95"],
                        color="#104281", alpha=0.2, lw=0)
        for spec, ls, color, label in [("coarse", "--", "#9c9a92", "H0, 4 popularity bins"),
                                       ("fine", ":", "#eb6834", "H0, 8 popularity bins")]:
            n = d[d["spec"] == spec].sort_values("tenth")
            ax.plot(n["tenth"], n["null_gap"], color=color, ls=ls, lw=2, label=label)
        ax.axhline(0, color="#73726c", lw=0.8, ls=":")
        ax.set_title(g, loc="left", fontsize=9)
        ax.set_xticks(range(1, 11))
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Early v4 minus late v3,\nshare of ratings (pp)", fontsize=8)
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.supxlabel("Tenth of the user's rating history", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    pd.set_option("display.width", 250)
    obs = observed_counts()
    ok = not_depleted(obs.index)
    kept = obs[ok].reset_index().groupby(["cohort", "tenth"]).size().unstack("tenth")
    print(f"user-tenths kept with >= {MIN_UNRATED} unrated top-100 movies: {int(ok.sum()):,} of {len(obs):,}")
    print(kept.to_string(), "\n")
    frames = []
    for spec in BIN_SQLS:
        null = null_counts(load_user_bins(spec), load_strata(spec)).reindex(
            index=obs.index, columns=obs.columns, fill_value=0)
        keep = null.sum(axis=1) > 0
        for subset, mask in [("all users", keep), (f"unrated >= {MIN_UNRATED}", keep & ok)]:
            r = summarize(to_share(obs[mask]), to_share(null[mask]))
            r["spec"], r["subset"] = spec, subset
            frames.append(r)
    res = pd.concat(frames, ignore_index=True)
    show = res[(res["spec"] == "fine") & res["tenth"].isin([5, 10])]
    print(show[["subset", "genre", "tenth", "observed_gap", "null_gap", "excess_gap", "excess_ci95"]]
          .round(2).to_string(index=False))
    make_figure(res[res["subset"] == "all users"], ROOT / "figures/cp3_popularity_null.png")
    (ROOT / "reports/checkpoint3_popularity_null.json").write_text(json.dumps(res.to_dict("records"), indent=2))


if __name__ == "__main__":
    main()