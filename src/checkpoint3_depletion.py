"""Checkpoint 3, Finding 1: depletion diagnostic.

For each cohort user, at the first rating of every tenth of their history: how many of that month's top-100 movies
(trailing 12 months, as in checkpoint1.py) have they not rated yet?

Run after src/checkpoint1.py has built data/processed/eda.duckdb:
    python src/checkpoint3_depletion.py
Writes:
    figures/cp3_depletion.png
    reports/checkpoint3_depletion.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import checkpoint2_taste as taste

MIN_RATINGS = 100
COLORS = {"late v3": taste.ERA_COLORS["v3"], "early v4": taste.ERA_COLORS["v4+"]}


def load_unrated():
    """One row per (user, tenth): number of the month's top-100 movies the user has not rated before that tenth."""
    q = taste.users_sql(MIN_RATINGS) + f""",
t0 as (
    select s.userId, u.n, u.first_ts,
           floor((s.ord - 1) * 10.0 / u.n)::int + 1 as tenth, min(s.ord) as ord0
    from seqp s join u on u.userId = s.userId
    where u.first_ts >= epoch(timestamp '{taste.COHORT_START}') and u.first_ts < epoch(timestamp '{taste.COHORT_END}')
    group by all
),
t as (
    select t0.*, s0.ym as ym0
    from t0 join seqp s0 on s0.userId = t0.userId and s0.ord = t0.ord0
),
top_m as (select movieId, ym from pop_rank where trail12 > 0 and trail_rank <= 100),
top_n as (select ym, count(*) as n_top from top_m group by ym),
hit as (
    select t.userId, t.tenth, count(*) as rated_top
    from t
    join seqp prev on prev.userId = t.userId and prev.ord < t.ord0
    join top_m on top_m.movieId = prev.movieId and top_m.ym = t.ym0
    group by all
)
select t.userId, t.n, t.first_ts, t.tenth, top_n.n_top - coalesce(hit.rated_top, 0) as unrated_top100
from t
join top_n on top_n.ym = t.ym0
left join hit on hit.userId = t.userId and hit.tenth = t.tenth
"""
    return taste.run_sql(q)


def summarize(df):
    df = df.copy()
    df["cohort"] = np.where(pd.to_datetime(df["first_ts"], unit="s") < taste.V4, "late v3", "early v4")
    df["band"] = pd.cut(df["n"], [0, 299, 999, np.inf], labels=taste.BANDS)
    g = df.groupby(["cohort", "band", "tenth"], observed=True)["unrated_top100"]
    return g.agg(users="size", mean="mean", p10=lambda s: s.quantile(0.10)).reset_index()


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    for ax, band in zip(axes, taste.BANDS):
        for cohort, color in COLORS.items():
            d = res[(res["band"] == band) & (res["cohort"] == cohort)].sort_values("tenth")
            ax.plot(d["tenth"], d["mean"], color=color, lw=2, marker="o", ms=3, label=f"{cohort} mean")
            ax.plot(d["tenth"], d["p10"], color=color, lw=1.5, ls=":", label=f"{cohort} 10th pct")
        ax.set_title(f"Users with {band} ratings", loc="left", fontsize=8.5)
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 105)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Top-100 movies not yet rated", fontsize=8)
    axes[0].legend(frameon=False, fontsize=6)
    fig.supxlabel("Tenth of the user's rating history (at its first rating)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    pd.set_option("display.width", 250)
    res = summarize(load_unrated())
    show = res[res["tenth"].isin([1, 5, 10])]
    print(show.round(1).to_string(index=False))
    make_figure(res, ROOT / "figures/cp3_depletion.png")
    (ROOT / "reports/checkpoint3_depletion.json").write_text(json.dumps(res.to_dict("records"), indent=2, default=str))


if __name__ == "__main__":
    main()