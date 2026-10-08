# Key numbers for the proposal

All values below come from the downloaded CSVs and the IMDb snapshot, not from hand entry into an analysis. Run [the notebook](../notebooks/01_dataset_orientation.ipynb) to rebuild `reports/eda_metrics.json`, `reports/external_metrics.json`, and the figures. Percentages below divide the named count by its stated denominator and round to one decimal place. UTC is used for every timestamp.

| Claim | Value | Producing code / metric |
|---|---:|---|
| MovieLens ratings | 32,000,204 | `src/eda.py`, `analyze()` → `sizes.ratings` |
| Users represented | 200,948 | `src/eda.py` → `sizes.users` |
| Catalog entries | 87,585 | `src/eda.py` → `sizes.movies` |
| Tag applications | 2,000,072 | `src/eda.py` → `sizes.tags` |
| Entries with ratings | 84,432 = 87,585 − 3,153 | `src/eda.py` → `sizes.movies`, `quality.movies_without_rating` |
| User–movie duplicate pairs | 0 | `src/eda.py` → `quality.duplicate_pairs` |
| Invalid scores | 0 | `src/eda.py` → `quality.invalid_ratings` |
| Rated IDs missing from movies | 0 | `src/eda.py` → `quality.ratings_missing_movie` |
| Tag-only entries | 3,153 | `src/eda.py` → `quality.tag_only_movies` |
| Title year parse failures | 617 | `src/eda.py` → `quality.missing_title_year` |
| Entries without genre labels | 7,080 | `src/eda.py` → `quality.no_genres` |
| Users with 20–25 ratings | 23,039 (11.5% of users) | `src/eda.py` → `user_activity.users_20_to_25`, `sizes.users` |
| User rating count median / maximum | 73 / 33,332 | `src/eda.py` → `user_activity.median`, `user_activity.max` |
| Rated movie count median / maximum | 5 / 102,929 | `src/eda.py` → `movie_activity.median`, `movie_activity.max` |
| Rated movies with <5 ratings | 40,548 (48.0% of rated entries) | `src/eda.py` → `movie_activity.under_5`; denominator 84,432 |
| Rated movies with <20 ratings | 61,082 (72.3%) | `src/eda.py` → `movie_activity.under_20`; denominator 84,432 |
| Rated movies with <100 ratings | 72,241 (85.6%) | `src/eda.py` → `movie_activity.under_100`; denominator 84,432 |
| Ratings held by top 1% / 10% of rated entries | 54.9% / 95.7% | `src/eda.py` → `concentration.top_1_share`, `top_10_share` |
| Gini of movie rating counts | 0.952 | `src/eda.py` → `concentration.gini` |
| Ratings before August 1997 | 2,159,894 across 33,358 users | `src/eda.py` → `rating_scale.pre_aug_1997_*` |
| Ratings before September 1997 | 2,159,894 across 33,358 users | `src/eda.py` → `rating_scale.pre_sep_1997_*` |
| Pre-v3 half-star ratings | 0 | `src/eda.py` → `rating_scale.pre_v3_half_stars` |
| Ratings predating parsed release year | 1,014 on 100 entries | `src/eda.py` → `pre_release` |
| Median gap between a user's adjacent ratings | 10 seconds | `src/eda.py`, `sessions()` → `gaps.median_seconds` |
| Ratings in 50+ sessions, 30-minute rule | 21,347,597 (66.7% of ratings) | `src/eda.py` → `sessions_30.ratings_in_50plus`, `sizes.ratings` |
| Ratings in 50+ sessions, 60-minute rule | 21,668,200 (67.7% of ratings) | `src/eda.py` → `sessions_60.ratings_in_50plus`, `sizes.ratings` |
| Median first-session size, 30/60-minute rules | 47 / 48 ratings | `src/eda.py` → `first_session_30.median_size`, `first_session_60.median_size` |
| First-session ratings, 30-minute rule | 16,166,278 | `src/eda.py` → `first_session_composition_30.ratings` |
| First-session prior-year top-1% share, 30/60-minute rules | 23.9% / 23.8% | `src/eda.py` → `first_session_composition_30.prior_top1_share`, `first_session_composition_60.prior_top1_share` |
| Users in post-1998 pre-v3 / v3 / v4+ cohorts | 35,717 / 62,900 / 67,933 | `src/eda.py` → `first20_prior_stable.users` |
| First-20 share in previous-year top 1%, same cohorts | 8.58% / 8.58% / 66.05% | `src/eda.py` → `first20_prior_stable.prior_top1_share` |
| First-20 share in whole-sample top 1%, same cohorts | 63.04% / 63.52% / 85.02% | `src/eda.py` → `first20_global_stable.top1_share` |
| October 2014 first users / prior-year top-1% first-20 share | 267 / 15.9% | `src/eda.py` → `first20_monthly` row `2014-10` |
| November 2014 first users / prior-year top-1% first-20 share | 435 / 57.1% | `src/eda.py` → `first20_monthly` row `2014-11` |
| Stable users with defined mainstreamness score | 166,037 | `src/eda.py` → `mainstream.users` |
| Median mainstreamness score | 0.963 | `src/eda.py` → `mainstream.median` |
| Movies first rated in 2008 / 2009 / 2015 | 933 / 1,910 / 7,798 | `src/eda.py` → `catalog` |
| Matched IMDb title records / vote counts | 87,355 / 87,226 | `src/external.py` → `match.basics_matched`, `match.votes_matched` |
| Titles with both MovieLens and IMDb counts | 84,108 | `src/external.py` → `match.dual_popularity` |
| Correlation of log MovieLens and IMDb counts | 0.850 | `src/external.py` → `correlation.pearson_log` |
| Mixed-popularity quadrant sizes | 2,003 / 2,002 | `src/external.py` → `quadrants` |
| Matched entries with IMDb type other than movie or TV movie | 10,058 | `src/external.py` → `match.non_movie_types` |
| Top 1% of taggers' share | 65.6% | `src/eda.py` → `tag_concentration.top1_users_tag_share` |

The four CSV MD5 values are checked against the [README table](https://files.grouplens.org/datasets/movielens/ml-32m-README.html#verifying-the-dataset-contents) before analysis. The full underlying metrics and example records are in `reports/*.json` and are produced by the named code.
