"""Checkpoint 2: does a user's first-20 behavior predict whether they stay? (after Hamilton et al. 2017)

Hamilton et al. found that a user's future loyalty is visible in their first interactions. We ask the same of
MovieLens, separately for users who joined in the two years before v4 and the two years after (Finding 3 windows).

Outcome: "stays" = still rating at least 365 days after their 20th rating. Measuring from the 20th rating means
no feature (all computed from ratings 1-20) can contain the outcome.

Run after src/checkpoint1.py has built data/processed/eda.duckdb (table `seqp`):
    python src/checkpoint2_retention.py
Writes:
    reports/checkpoint2_retention.json
    figures/cp2_retention.png
"""
import json
import sys
import warnings
from pathlib import Path

# Spurious matmul RuntimeWarnings (numpy 2 + macOS Accelerate) and scipy's "unknown option iprint": predictions were
# checked against a plain einsum recomputation (max diff 4e-16) and lbfgs converges in < 10 iterations.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul")
warnings.filterwarnings("ignore", message="Unknown solver options")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import eda  # noqa: E402  (also puts .deps on sys.path)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

V4 = "2014-11-01"
WINDOW_START, WINDOW_END = "2012-11-01", "2016-11-01"  # same windows as Finding 3
COHORTS = {"late v3": "#eb6834", "early v4": "#1baf7a"}  # same colors as checkpoint1.py
FEATURES = {  # all computed from ratings 1-20 only
    "pop_share": "Popular share (top-100)",
    "mean_rating": "Mean rating",
    "sd_rating": "Rating spread (SD)",
    "genre_entropy": "Genre variety (entropy)",
    "log_span_hours": "Time from 1st to 20th rating (log)",
    "sessions": "Number of sittings",
}
N_BOOT = 200
OUT = ROOT / "reports/checkpoint2_retention.json"


def load_users(con):
    df = con.execute(f"""
      WITH u AS (SELECT userId, min(ts) first_ts, max(ts) last_ts FROM seqp GROUP BY 1
                 HAVING min(ts) >= epoch(TIMESTAMP '{WINDOW_START}') AND min(ts) < epoch(TIMESTAMP '{WINDOW_END}')),
      f AS (SELECT s.* FROM seqp s JOIN u USING (userId) WHERE s.ord <= 20),
      g AS (SELECT userId, genre, count(*) c FROM
              (SELECT f.userId, unnest(string_split(m.genres, '|')) genre FROM f JOIN movies m USING (movieId))
            WHERE genre NOT IN ('(no genres listed)', 'IMAX') GROUP BY 1, 2),
      ent AS (SELECT userId, -sum(p * ln(p)) genre_entropy FROM
                (SELECT userId, c / sum(c) OVER (PARTITION BY userId) p FROM g) GROUP BY 1)
      SELECT f.userId, count(*) n20, avg(top100_trail) pop_share, avg(rating) mean_rating,
        stddev_pop(rating) sd_rating, ln(1 + (max(ts) - min(ts)) / 3600.0) log_span_hours,
        count(DISTINCT sid) sessions, max(ts) t20, any_value(u.first_ts) first_ts, any_value(u.last_ts) last_ts,
        coalesce(any_value(ent.genre_entropy), 0) genre_entropy
      FROM f JOIN u USING (userId) LEFT JOIN ent USING (userId) GROUP BY f.userId""").df()
    assert (df.n20 == 20).all(), "every user must contribute exactly 20 ratings"  # guards against leakage
    df["cohort"] = np.where(pd.to_datetime(df.first_ts, unit="s") < V4, "late v3", "early v4")
    df["stays"] = ((df.last_ts - df.t20) >= 365 * 86400).astype(int)
    return df


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def fit_cohort(d, rng):
    X, y = d[list(FEATURES)].to_numpy(float), d.stays.to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    auc = cross_val_score(model(), X, y, cv=cv, scoring="roc_auc")
    auc_pop = cross_val_score(model(), X[:, [0]], y, cv=cv, scoring="roc_auc")
    coef = model().fit(X, y)[-1].coef_[0]
    # ponytail: percentile bootstrap, 200 draws; enough for 95% CIs at these sample sizes
    boot = np.array([model().fit(X[i], y[i])[-1].coef_[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    terc = pd.qcut(d.pop_share.rank(method="first"), 3, labels=["least popular third", "middle third", "most popular third"])
    return {
        "users": len(d), "stay_rate": float(y.mean()),
        "cv_auc": float(auc.mean()), "cv_auc_sd": float(auc.std()),
        "cv_auc_pop_only": float(auc_pop.mean()),
        "odds_ratio_per_sd": {k: {"or": float(np.exp(c)), "lo": float(np.exp(a)), "hi": float(np.exp(b))}
                              for k, c, a, b in zip(FEATURES, coef, lo, hi)},
        "stay_rate_by_pop_third": d.groupby(terc, observed=True).stays.mean().to_dict(),
        "feature_means": d[list(FEATURES)].mean().to_dict(),
    }


def cross_auc(train, test):
    m = model().fit(train[list(FEATURES)].to_numpy(float), train.stays)
    return float(roc_auc_score(test.stays, m.predict_proba(test[list(FEATURES)].to_numpy(float))[:, 1]))


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": .6, "axes.axisbelow": True})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.5, 1], "wspace": .35})
    ax = axes[0]
    for i, (c, color) in enumerate(COHORTS.items()):
        o = out["cohorts"][c]["odds_ratio_per_sd"]
        ys = np.arange(len(FEATURES)) + (i - .5) * .3
        vals = [o[k]["or"] for k in FEATURES]
        err = [[v - o[k]["lo"] for k, v in zip(FEATURES, vals)], [o[k]["hi"] - v for k, v in zip(FEATURES, vals)]]
        ax.errorbar(vals, ys, xerr=err, fmt="o", ms=5, color=color, capsize=2, label=c)
    ax.axvline(1, color="#73726c", lw=.8, ls=":")
    ax.set_xscale("log"); ax.set_xticks([.6, .8, 1, 1.25, 1.6], ["0.6", "0.8", "1", "1.25", "1.6"])
    ax.set_yticks(range(len(FEATURES)), FEATURES.values(), fontsize=7.5); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Odds of staying, per 1 SD increase (95% CI)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(.4, 1.0))
    ax = axes[1]
    thirds = ["least popular third", "middle third", "most popular third"]
    for i, (c, color) in enumerate(COHORTS.items()):
        r = out["cohorts"][c]["stay_rate_by_pop_third"]
        ax.bar(np.arange(3) + (i - .5) * .38, [r[t] for t in thirds], width=.36, color=color)
    ax.set_xticks(range(3), ["Least\npopular", "Middle", "Most\npopular"]); ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_ylabel("Share still rating a year later"); ax.set_xlabel("Third of first-20 popular share")
    fig.savefig(ROOT / "figures/cp2_retention.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    con = eda.connect()
    eda.prepare(con)
    df = load_users(con)
    con.close()
    rng = np.random.default_rng(0)
    a, b = df[df.cohort == "late v3"], df[df.cohort == "early v4"]
    out = {"outcome": "still rating >= 365 days after 20th rating", "windows": [WINDOW_START, V4, WINDOW_END],
           "cohorts": {"late v3": fit_cohort(a, rng), "early v4": fit_cohort(b, rng)},
           "cross_cohort_auc": {"train v3 -> test v4": cross_auc(a, b), "train v4 -> test v3": cross_auc(b, a)}}
    OUT.write_text(json.dumps(out, indent=2, default=str))
    figure(out)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
