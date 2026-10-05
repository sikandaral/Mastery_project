# Project outline: MovieLens as a platform that creates its own data

**Working question.** How did documented MovieLens changes in onboarding, list ordering, and catalog policy coincide with changes in the concentration of users' ratings on already popular films? Are apparently niche users stable across eras? A later secondary analysis can ask whether recommenders trained on these records intensify concentration.

**Why this is distinct.** [Abdollahpouri et al. (2019)](https://arxiv.org/abs/1907.13286) documented recommender popularity bias, including for niche users. This project investigates the historical site and feedback loop that helped produce the training data, using [Harper and Konstan's platform history](https://doi.org/10.1145/2827872) as its timeline.

| Stage | Work | Output / decision |
|---|---|---|
| 1. Orientation (October 5–8) | Verify MovieLens 32M, describe users/items/sessions, check anomalies, join external popularity. | Five-page proposal, EDA notebook, 6–8 figures, key-number table, open questions. |
| 2. Historical measures | Build previous-year and previous-month popularity, first-20 and first-session concentration, fixed-catalog sensitivity. | Cohort trends with denominators and uncertainty intervals. |
| 3. Platform event analysis | Compare nearby monthly cohorts at v3 (February 18, 2003), catalog opening (September 2008), and v4 (November 2014); account for pre-trends and changing film mix. | Descriptive event plots and qualified interpretations. |
| 4. User stability | Follow users who rate across periods; compare within-user mainstreamness and activity changes. | Stability and attrition analysis with minimum-observation rules. |
| 5. External context | Use IMDb vote counts as a current cross-platform check; restrict by film age. Obtain production language/country metadata only if a reliable source is available. | Match-rate analysis and on-site versus external popularity examples. |
| 6. Optional recommender baseline | After historical findings are stable, compare a popularity baseline and one standard recommender using chronological splits and exposure-concentration metrics. | Evidence on amplification, explicitly secondary to the platform question. |

**Primary unit and denominator.** The principal unit is a user's first 20 rated movies. Popularity is the movie's MovieLens count at the end of the previous year; a “top 1%” movie is ranked among movies already rated by then. Report users, films, and unrated-before share for every cohort. Exclude pre-1998 first users from primary event comparisons while investigating the pre-launch timestamp anomaly.

**Interpretation rule.** Treat observed changes as associations. The file contains no impression logs, displayed rankings, demographics, or counterfactual exposures. MovieLens timestamps record rating entry, not viewing.

**Immediate submission package.** [Proposal PDF](project_proposal.pdf), [proposal source](project_proposal.md), [findings](eda_findings.md), [key numbers](key_numbers.md), [open questions](open_questions.md), [notebook](../notebooks/01_dataset_orientation.ipynb), and [figures](../figures/captions.md).
