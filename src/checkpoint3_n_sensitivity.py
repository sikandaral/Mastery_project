"""Checkpoint 3: how sensitive are the CP2 Finding 1 genre gaps to the minimum-ratings threshold n?

Run after src/checkpoint1.py has built data/processed/eda.duckdb:
    python src/checkpoint3_n_sensitivity.py
Writes:
    figures/cp3_n_sensitivity.png
    reports/checkpoint3_n_sensitivity.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import checkpoint2_taste as taste

THRESHOLDS = (40, 60, 100)
FOCUS_GENRES = ["Comedy", "Action", "Crime"]  # the three genres in Figure 1 of CP2
COLORS = {"Comedy": "#2a78d6", "Action": "#eb6834", "Crime": "#1baf7a"}
TENTHS_SHOWN = (1, 10)


def cohort_gaps(thresholds):
    """Pooled early v4 minus late v3 gap (percentage points) in genre share, for each threshold n."""
    counts = taste.load_cohort_counts(min(thresholds))  # load once at the lowest n, filter below
    rows = []
    for t in thresholds:
        mean, ci, n = taste.mean_shares(counts[counts["n"] >= t])
        for g in FOCUS_GENRES:
            for tenth in TENTHS_SHOWN:
                late, early = ("late v3", tenth), ("early v4", tenth)
                rows.append({
                    "n_min": t, "genre": g, "tenth": tenth,
                    "gap_pp": float(mean.loc[early, g] - mean.loc[late, g]),
                    "ci95": float(np.hypot(ci.loc[early, g], ci.loc[late, g])),
                    "users_late_v3": int(n.loc[late]), "users_early_v4": int(n.loc[early]),
                })
    return pd.DataFrame(rows)


def band_gaps(low):
    """v4+ minus v3 gap by rating-volume band: the new band below 100 ratings vs the CP2 100-299 band."""
    stats = taste.load_lifecycle_stats(low)
    rows = []
    for band in [f"{low}-99", "100-299"]:
        for g in FOCUS_GENRES:
            for tenth in TENTHS_SHOWN:
                s = stats[(stats["band"] == band) & (stats["genre"] == g) & (stats["tenth"] == tenth)].set_index("era")
                rows.append({
                    "band": band, "genre": g, "tenth": tenth,
                    "gap_pp": float(s.loc["v4+", "mean"] - s.loc["v3", "mean"]),
                    "ci95": float(np.hypot(s.loc["v4+", "ci95"], s.loc["v3", "ci95"])),
                    "users_v3": int(s.loc["v3", "n_users"]), "users_v4plus": int(s.loc["v4+", "n_users"]),
                })
    return pd.DataFrame(rows)


def make_figure(cg, bg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for i, g in enumerate(FOCUS_GENRES):
        off = (i - 1) * 0.2
        d = cg[(cg["genre"] == g) & (cg["tenth"] == 1)].sort_values("n_min")
        a.errorbar(np.arange(len(d)) + off, d["gap_pp"], yerr=d["ci95"], fmt="o", ms=4, capsize=2,
                   color=COLORS[g], label=g)
        e = bg[(bg["genre"] == g) & (bg["tenth"] == 1)]  # rows are already in band order
        b.errorbar(np.arange(len(e)) + off, e["gap_pp"], yerr=e["ci95"], fmt="o", ms=4, capsize=2, color=COLORS[g])
    a.set_xticks(range(len(THRESHOLDS)), [f"n ≥ {t}" for t in THRESHOLDS])
    b.set_xticks(range(2), [f"{x} ratings" for x in bg["band"].unique()])
    a.set_title("A. Late v3 vs early v4 (pooled)", loc="left", fontsize=8.5)
    b.set_title("B. v3 vs v4+, by rating-volume band", loc="left", fontsize=8.5)
    a.set_ylabel("Gap in first-tenth genre share (pp)", fontsize=8)
    a.legend(frameon=False, fontsize=7)
    for ax in (a, b):
        ax.axhline(0, color="#73726c", lw=0.8, ls=":")
        ax.set_xlim(-0.6, len(THRESHOLDS) - 0.4 if ax is a else 1.6)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    pd.set_option("display.width", 250)
    cg = cohort_gaps(THRESHOLDS)
    bg = band_gaps(min(THRESHOLDS))
    print("A. early v4 minus late v3 (pp), by minimum ratings n:")
    print(cg.round(2).to_string(index=False))
    print("\nB. v4+ minus v3 (pp), by rating-volume band:")
    print(bg.round(2).to_string(index=False))
    (ROOT / "figures").mkdir(exist_ok=True)
    make_figure(cg, bg, ROOT / "figures/cp3_n_sensitivity.png")
    out = {"thresholds": list(THRESHOLDS), "cohort_gaps": cg.to_dict("records"), "band_gaps": bg.to_dict("records")}
    (ROOT / "reports/checkpoint3_n_sensitivity.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()