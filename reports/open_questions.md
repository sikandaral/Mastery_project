# Open questions and decisions

1. **Pre-launch timestamps.** The August 1997 landmark conflicts with 2,159,894 earlier rating timestamps. Harper and Konstan say the EachMovie seed ratings were not included in earlier MovieLens datasets. Ask GroupLens whether these dates reflect migrated/backdated records or an export issue. Exclude them from primary platform-era inference now.
2. **The v4 day.** The source documents November 2014, not an exact day. The code uses November 1 in UTC. A month-level event comparison and a one-month exclusion around that date are safer than claiming a daily discontinuity.
3. **Historical popularity definition.** The no-look-ahead score uses the previous calendar year's cumulative MovieLens count and top 1% of titles already rated at least once. Recheck results using previous-month counts, a fixed catalog, and a minimum prior-history threshold. The full-sample definition gives very different levels.
4. **Stable “niche” users.** A person's early and later ratings may differ because the catalog and site changed. Define stability within users who span eras and have enough observations in each era; report attrition and avoid labeling users with sparse early histories.
5. **Catalog entry proxy.** First rating is only an upper bound on when a title became available. Check whether tags precede first ratings and whether MovieLens can provide true listing dates.
6. **External snapshot mismatch.** IMDb votes were retrieved in 2026, whereas MovieLens stops in October 2023. Restrict comparisons to titles released well before 2023 and test release-year strata. IMDb alternative-title `language` and `region` cannot establish original language or production country. A TMDB key or another source with production fields is needed for that analysis; no key was available in the current environment.
7. **Movie type.** IMDb calls 10,058 matched MovieLens entries neither `movie` nor `tvMovie`. Decide whether to retain shorts, videos, and television entries in the primary catalog denominator, and show a sensitivity check.
8. **Rating scale and session behavior.** Half-star use does not shift smoothly after the documented v3 launch. Check monthly patterns and possible cohort composition. Keep 30- and 60-minute session definitions side by side.
9. **Possible release-date anomalies.** 1,014 ratings precede the year embedded in the title. Inspect the high-count examples before treating these as data errors; title year, advance ratings, or timing could differ.
10. **Causal scope.** Ratings reveal neither displayed recommendations nor unrated impressions. Phrase results as associations around documented design changes. A causal claim would require exposure logs, experiments, or another credible design.

## Team decisions for the October 8 proposal

- Make **onboarding and display design around v3/v4** the main question; treat catalog expansion and external popularity as supporting analyses.
- Use **post-1998 users** for the primary first-20 comparisons, while reporting excluded pre-launch/transition data prominently.
- Keep recommender amplification as a later secondary analysis. The prior [Abdollahpouri et al. study](https://arxiv.org/abs/1907.13286) already establishes a popularity-bias result on MovieLens 1M; the distinct contribution here is platform history.
