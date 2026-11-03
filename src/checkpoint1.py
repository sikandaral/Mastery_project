"""Checkpoint 1 analyses: popularity-definition robustness, observation windows,
behavioral user profiles, and a composition-vs-within decomposition of the v4 shift.

Run after src/eda.py has built data/processed/*.parquet:  python src/checkpoint1.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import eda  # noqa: E402  (also puts .deps on sys.path)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

V3, V4 = "2003-02-18", "2014-11-01"
OUT = ROOT / "reports/checkpoint1_metrics.json"


def build(con):
    """Persistent tables, built once: popularity known *before* each month, and each rating with flags."""
    # ponytail: tables cached in eda.duckdb; DROP them to rebuild.
    con.execute("""CREATE TABLE IF NOT EXISTS pop_rank AS
      WITH mc AS (SELECT movieId, date_trunc('month', to_timestamp(timestamp))::DATE ym, count(*) n
                  FROM ratings GROUP BY 1, 2),
      months AS (SELECT unnest(generate_series((SELECT min(ym) FROM mc), (SELECT max(ym) FROM mc),
                  INTERVAL 1 MONTH))::DATE ym),
      grid AS (SELECT m.movieId, mo.ym, coalesce(mc.n, 0) n
               FROM (SELECT DISTINCT movieId FROM mc) m CROSS JOIN months mo LEFT JOIN mc USING (movieId, ym)),
      pop AS (SELECT movieId, ym,
               sum(n) OVER (PARTITION BY movieId ORDER BY ym ROWS UNBOUNDED PRECEDING) - n cum_before,
               coalesce(sum(n) OVER (PARTITION BY movieId ORDER BY ym
                        ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING), 0) trail12
              FROM grid)
      SELECT movieId, ym, cum_before, trail12,
        row_number() OVER (PARTITION BY ym ORDER BY cum_before DESC, movieId) cum_rank,
        row_number() OVER (PARTITION BY ym ORDER BY trail12 DESC, movieId) trail_rank,
        count(*) FILTER (WHERE cum_before > 0) OVER (PARTITION BY ym) active
      FROM pop""")
    # m1 reproduces the proposal measure: top 1% by cumulative count at the end of the previous calendar year.
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='10GB'")
    con.execute("""CREATE TABLE IF NOT EXISTS seq AS
      WITH s AS (SELECT userId, movieId, rating, timestamp ts,
                   date_trunc('month', to_timestamp(timestamp))::DATE ym,
                   row_number() OVER (PARTITION BY userId ORDER BY timestamp, movieId) ord,
                   timestamp - lag(timestamp) OVER (PARTITION BY userId ORDER BY timestamp, movieId) gap
                 FROM ratings)
      SELECT userId, movieId, rating, ts, ym, ord::INT ord,
        sum(CASE WHEN gap IS NULL OR gap >= 1800 THEN 1 ELSE 0 END)
          OVER (PARTITION BY userId ORDER BY ord ROWS UNBOUNDED PRECEDING)::INT sid
      FROM s""")
    con.execute("""CREATE TABLE IF NOT EXISTS seqp AS
      SELECT s.*,
        (y.cum_before > 0 AND y.cum_rank <= ceil(y.active * .01))::INT m1_prior_year_top1pct,
        (p.cum_before > 0 AND p.cum_rank <= 100)::INT top100_cum,
        (p.trail12 > 0 AND p.trail_rank <= 100)::INT top100_trail,
        (p.cum_before > 0 AND p.cum_rank <= 1000)::INT top1000_cum,
        (p.cum_before = 0)::INT unrated_before,
        CASE WHEN p.cum_before > 0 THEN 1 - (p.cum_rank - 1) / p.active ELSE 0 END pct_cum
      FROM seq s JOIN pop_rank p USING (movieId, ym)
      JOIN pop_rank y ON y.movieId = s.movieId AND y.ym = date_trunc('year', s.ym)::DATE""")



WINDOWS = [(1, 10), (1, 20), (21, 50), (51, 100), (101, 200)]
MEASURE = "top100_trail"  # primary: top 100 movies by ratings in the previous 12 months


def user_table(con):
    win = ",\n".join(f"avg({MEASURE}) FILTER (WHERE ord BETWEEN {a} AND {b}) w_{a}_{b}" for a, b in WINDOWS)
    u = con.execute(f"""SELECT userId, count(*) n, min(ts) first_ts, max(ts) last_ts,
        count(*) FILTER (WHERE sid = 1) first_session_n,
        avg(m1_prior_year_top1pct) FILTER (WHERE ord <= 20) f20_m1,
        avg(top100_cum) FILTER (WHERE ord <= 20) f20_t100c,
        avg(top100_trail) FILTER (WHERE ord <= 20) f20_t100t,
        avg({MEASURE}) FILTER (WHERE sid = 1) w_first_session,
        {win},
        avg({MEASURE}) FILTER (WHERE ord > 20) later_main,
        count(*) FILTER (WHERE ord > 20) n_later,
        avg(rating) mean_rating
      FROM seqp GROUP BY userId""").df()
    u["start"] = pd.to_datetime(u.first_ts, unit="s")
    u["era"] = np.select([u.start < V3, u.start < V4], ["pre-v3", "v3"], "v4+")
    u["tenure_days"] = (u.last_ts - u.first_ts) / 86400
    u["engagement"] = pd.cut(u.tenure_days, [-1, 1, 30, 365, np.inf],
                             labels=["one sitting (<1 day)", "short-lived (1-30 days)",
                                     "returning (1-12 months)", "long-term (1+ year)"]).astype(str)
    u = u[u.start >= "1998-01-01"].copy()  # pre-1998 timestamps unresolved (see open_questions.md)
    # ponytail: taste = pooled tertiles of later mainstream share; needs >=20 later ratings.
    ok = u.n_later >= 20
    lo, hi = u.loc[ok, "later_main"].quantile([1 / 3, 2 / 3])
    u["taste"] = np.where(~ok, "too few later ratings (<40 total)",
                          np.where(u.later_main >= hi, "mainstream", np.where(u.later_main < lo, "niche", "mixed")))
    return u, {"taste_tertile_cuts": [lo, hi]}


def mean_ci(s):
    s = s.dropna()
    return {"n": int(len(s)), "mean": float(s.mean()), "ci95": float(1.96 * s.std() / np.sqrt(len(s))) if len(s) > 1 else None}


def decompose(u, outcome):
    """Shift-share with average weights: total = composition + within (exact)."""
    a = u[(u.start >= "2012-11-01") & (u.start < V4)]
    b = u[(u.start >= V4) & (u.start < "2016-11-01")]
    cells = ["engagement", "taste"]
    sa, sb = a.groupby(cells).size() / len(a), b.groupby(cells).size() / len(b)
    ya, yb = a.groupby(cells)[outcome].mean(), b.groupby(cells)[outcome].mean()
    t = pd.DataFrame({"share_v3late": sa, "share_v4early": sb, "y_v3late": ya, "y_v4early": yb}).fillna(0)
    t["composition"] = (t.share_v4early - t.share_v3late) * (t.y_v3late + t.y_v4early) / 2
    t["within"] = (t.y_v4early - t.y_v3late) * (t.share_v3late + t.share_v4early) / 2
    total = b[outcome].mean() - a[outcome].mean()
    assert abs(total - t.composition.sum() - t.within.sum()) < 1e-9
    by = {d: pd.DataFrame({"v3late": a.groupby(d)[outcome].mean(), "v4early": b.groupby(d)[outcome].mean()})
          .reset_index().to_dict("records") for d in cells}
    return {"by_profile": by, "users_v3late": len(a), "users_v4early": len(b), "y_v3late": a[outcome].mean(), "y_v4early": b[outcome].mean(),
            "total": total, "composition": t.composition.sum(), "within": t.within.sum(),
            "cells": t.reset_index().to_dict("records")}


def lifecycle(con):
    return con.execute(f"""WITH u AS (SELECT userId, count(*) n, min(ts) f FROM seqp GROUP BY 1
                                HAVING count(*) >= 100 AND min(ts) >= epoch(TIMESTAMP '1998-01-01')),
      ub AS (SELECT s.userId, u.n, u.f, floor((s.ord - 1) * 10.0 / u.n)::INT bin,
               avg({MEASURE}) main, avg(pct_cum) pct FROM seqp s JOIN u USING (userId) GROUP BY 1, 2, 3, 4)
      SELECT CASE WHEN f < epoch(TIMESTAMP '{V3}') THEN 'pre-v3' WHEN f < epoch(TIMESTAMP '{V4}') THEN 'v3' ELSE 'v4+' END era,
        CASE WHEN n < 300 THEN '100-299' WHEN n < 1000 THEN '300-999' ELSE '1000+' END band,
        bin, count(*) users, avg(main) main, avg(pct) pct
      FROM ub GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""").df()


def analyze(con):
    u, out = user_table(con)
    eras = ["pre-v3", "v3", "v4+"]
    out["era_users"] = u.era.value_counts().to_dict()
    out["first20_by_era"] = {e: {m: mean_ci(u.loc[u.era == e, m]) for m in ["f20_m1", "f20_t100c", "f20_t100t"]} for e in eras}
    ev = u[(u.start >= "2012-01-01") & (u.start < "2018-01-01")]
    out["monthly"] = [{"month": str(k.date()), **{m: mean_ci(g[m]) for m in ["f20_m1", "f20_t100c", "f20_t100t"]}}
                      for k, g in ev.groupby(ev.start.dt.to_period("M").dt.to_timestamp())]
    out["yearly"] = [{"year": int(k), "users": len(g), **{m: float(g[m].mean()) for m in ["f20_m1", "f20_t100c", "f20_t100t"]}}
                     for k, g in u.groupby(u.start.dt.year)]
    cols = ["w_1_10", "w_1_20", "w_first_session"] + [f"w_{a}_{b}" for a, b in WINDOWS[2:]]
    out["windows_fixed_users_200plus"] = {e: {c: mean_ci(u.loc[(u.era == e) & (u.n >= 200), c]) for c in cols} for e in eras}
    out["windows_all_available"] = {e: {c: mean_ci(u.loc[u.era == e, c]) for c in cols} for e in eras}
    out["first_session_median"] = u.groupby("era").first_session_n.median().to_dict()
    out["share_20_25"] = u.groupby("era").n.apply(lambda s: float((s <= 25).mean())).to_dict()
    for dim in ["engagement", "taste"]:
        out[f"{dim}_by_era"] = (pd.crosstab(u[dim], u.era, normalize="columns")).to_dict()
        out[f"{dim}_profile_stats"] = u.groupby(dim).agg(users=("userId", "size"), median_n=("n", "median"),
            median_first_session=("first_session_n", "median"), f20_t100t=("f20_t100t", "mean"),
            later_main=("later_main", "mean"), mean_rating=("mean_rating", "mean")).reset_index().to_dict("records")
    out["decomp_t100t"] = decompose(u, "f20_t100t")
    out["decomp_m1"] = decompose(u, "f20_m1")
    out["lifecycle"] = lifecycle(con).to_dict("records")
    out["first10_titles"] = con.execute(f"""WITH u AS (SELECT userId, min(ts) f FROM seq GROUP BY 1
                                 HAVING min(ts) >= epoch(TIMESTAMP '1998-01-01')),
      e AS (SELECT userId, CASE WHEN f < epoch(TIMESTAMP '{V3}') THEN 'pre-v3'
                 WHEN f < epoch(TIMESTAMP '{V4}') THEN 'v3' ELSE 'v4+' END era FROM u)
      SELECT era, title, count(*) / max(era_users) share_of_users FROM seq JOIN e USING (userId) JOIN movies USING (movieId)
      JOIN (SELECT era, count(*) era_users FROM e GROUP BY 1) USING (era) WHERE ord <= 10 GROUP BY 1, 2
      QUALIFY row_number() OVER (PARTITION BY era ORDER BY count(*) DESC) <= 5 ORDER BY 1, 3 DESC""").df().to_dict("records")
    return out


ERA_COLORS = {"pre-v3": "#2a78d6", "v3": "#eb6834", "v4+": "#1baf7a"}  # validated: dataviz validate_palette.js
MEASURES = {"f20_m1": ("prior-year top 1% (proposal)", "#104281", "-"),
            "f20_t100c": ("top 100, cumulative", "#2a78d6", "--"),
            "f20_t100t": ("top 100, trailing 12 months", "#86b6ef", "-")}


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                         "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": .6, "axes.axisbelow": True})
    fig_dir = ROOT / "figures"
    pct = lambda ax: ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))

    def vline(ax, x, label):
        ax.axvline(x, color="#73726c", lw=.8, ls=":")
        ax.text(x, ax.get_ylim()[1], f" {label}", fontsize=7, color="#52514e", va="top")

    # 1. Event study
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1.25], hspace=.08, wspace=.25)
    ax = fig.add_subplot(gs[:, 0])
    y = pd.DataFrame(out["yearly"])
    for m, (lab, c, ls) in MEASURES.items():
        ax.plot(y.year, y[m], color=c, ls=ls, lw=2, marker="o", ms=3, label=lab)
    ax.set_ylim(0, .85); pct(ax); vline(ax, 2003.13, "v3"); vline(ax, 2014.83, "v4")
    ax.set_title("(a) Yearly cohorts, 1998-2023", loc="left", fontsize=9)
    ax.set_xlabel("Year of first rating"); ax.set_ylabel("Share of first 20 ratings on popular movies")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, .93))
    mo = pd.DataFrame([{"month": pd.Timestamp(r["month"]), **{m: r[m]["mean"] for m in MEASURES},
                        **{m + "_ci": r[m]["ci95"] for m in MEASURES}, "n": r["f20_m1"]["n"]} for r in out["monthly"]])
    ax = fig.add_subplot(gs[0, 1]); axn = fig.add_subplot(gs[1, 1], sharex=ax)
    for m, (lab, c, ls) in MEASURES.items():
        ax.plot(mo.month, mo[m], color=c, ls=ls, lw=1.6)
        ax.fill_between(mo.month, mo[m] - mo[m + "_ci"], mo[m] + mo[m + "_ci"], color=c, alpha=.18, lw=0)
    ax.set_ylim(0, .85); pct(ax); vline(ax, pd.Timestamp(V4), "v4 (Nov 2014)")
    ax.set_title("(b) Monthly cohorts, 2012-2017, with 95% CI", loc="left", fontsize=9)
    plt.setp(ax.get_xticklabels(), visible=False)
    axn.bar(mo.month, mo.n, width=25, color="#9c9a92"); axn.set_ylabel("New users", fontsize=7)
    axn.axvline(pd.Timestamp(V4), color="#73726c", lw=.8, ls=":")
    fig.savefig(fig_dir / "cp1_01_event_study.png", bbox_inches="tight"); plt.close(fig)

    # 2. Observation windows (fixed users with 200+ ratings)
    cols = {"w_1_10": "1-10", "w_1_20": "1-20", "w_21_50": "21-50", "w_51_100": "51-100", "w_101_200": "101-200"}
    fig, ax = plt.subplots(figsize=(5.6, 3))
    for e, c in ERA_COLORS.items():
        w = out["windows_fixed_users_200plus"][e]
        ys = [w[k]["mean"] for k in cols]
        ax.errorbar(range(len(cols)), ys, yerr=[w[k]["ci95"] for k in cols], color=c, lw=2, marker="o", ms=5, capsize=2)
        ax.text(len(cols) - .9, ys[-1], f"{e} (n={w['w_1_20']['n']:,})", color="#30302e", fontsize=7.5, va="center")
    ax.set_xticks(range(len(cols)), cols.values()); ax.set_xlim(-.3, len(cols) + .4)
    ax.set_xlabel("Position in the user's own rating sequence"); pct(ax)
    ax.set_ylabel("Share on top-100 movies\n(trailing 12 months)")
    fig.savefig(fig_dir / "cp1_02_windows.png", bbox_inches="tight"); plt.close(fig)

    # 3. Profile prevalence by era
    short = lambda s: s.replace("too few later ratings (<40 total)", "<40 ratings total")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": .8})
    for ax, dim, order in [(axes[0], "engagement", ["one sitting (<1 day)", "short-lived (1-30 days)", "returning (1-12 months)", "long-term (1+ year)"]),
                           (axes[1], "taste", ["mainstream", "mixed", "niche", "too few later ratings (<40 total)"])]:
        d = out[f"{dim}_by_era"]
        for i, (e, c) in enumerate(ERA_COLORS.items()):
            ax.barh([j + (i - 1) * .27 for j in range(4)], [d[e][k] for k in order], height=.25, color=c, label=e)
        ax.set_yticks(range(4), [short(o) for o in order]); ax.invert_yaxis()
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.set_title(f"{dim.capitalize()} profile", loc="left", fontsize=9); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Share of era's users")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.08))
    fig.savefig(fig_dir / "cp1_03_profiles.png", bbox_inches="tight"); plt.close(fig)

    # 4. Decomposition
    dc = out["decomp_t100t"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1, 1.5], "wspace": .95})
    ax = axes[0]; vals = [dc["total"], dc["composition"], dc["within"]]
    ax.bar(["Total", "Who\njoined", "Within\ntypes"], vals, color=["#9c9a92", "#eb6834", "#1baf7a"], width=.6)
    for i, v in enumerate(vals):
        ax.text(i, max(v, 0) + .015, f"{100 * v:+.1f}", ha="center", fontsize=8)
    ax.axhline(0, color="#30302e", lw=.8); ax.set_ylim(-.03, .55); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Change, late v3 to early v4\n(percentage points)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    ax = axes[1]; rows = dc["by_profile"]["engagement"] + dc["by_profile"]["taste"]
    labels = [short(r.get("engagement", r.get("taste"))) for r in rows]
    for j, r in enumerate(rows):
        ax.plot([r["v3late"], r["v4early"]], [j, j], color="#c3c2b7", lw=1.5)
    ax.scatter([r["v3late"] for r in rows], range(len(rows)), color=ERA_COLORS["v3"], s=28, zorder=3, label="late v3 (Nov 2012-Oct 2014)")
    ax.scatter([r["v4early"] for r in rows], range(len(rows)), color=ERA_COLORS["v4+"], s=28, zorder=3, label="early v4 (Nov 2014-Oct 2016)")
    ax.axhline(3.5, color="#e6e6e3", lw=1); ax.set_yticks(range(len(rows)), labels, fontsize=7.5); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0)); ax.set_xlim(0, .8)
    ax.set_xlabel("First-20 share on top-100 movies")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(.4, 1.0))
    ax.grid(axis="y", visible=False)
    fig.savefig(fig_dir / "cp1_04_decomposition.png", bbox_inches="tight"); plt.close(fig)

    # 5. Lifecycle
    lc = pd.DataFrame(out["lifecycle"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, band in zip(axes, ["100-299", "300-999", "1000+"]):
        for e, c in ERA_COLORS.items():
            s = lc[(lc.band == band) & (lc.era == e)]
            ax.plot(s.bin + 1, s.main, color=c, lw=2, marker="o", ms=3, label=f"{e} (n={s.users.max():,})")
        ax.set_title(f"Users with {band} ratings", loc="left", fontsize=8.5); ax.set_xticks(range(1, 11))
        ax.set_xlabel("Tenth of user's rating history"); ax.legend(frameon=False, fontsize=6.5)
    axes[0].set_ylabel("Share on top-100 movies\n(trailing 12 months)"); pct(axes[0])
    fig.savefig(fig_dir / "cp1_05_lifecycle.png", bbox_inches="tight"); plt.close(fig)

def main():
    os.chdir(ROOT)
    con = eda.connect()
    eda.prepare(con)
    build(con)
    out = analyze(con)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    con.close()
    figures(out)
    return out


if __name__ == "__main__":
    main()
