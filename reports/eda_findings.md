# MovieLens 32M: dataset orientation and EDA findings

## Dataset tl;dr

The verified MovieLens 32M files contain **32,000,204 valid, unique user–movie ratings** from **200,948 users** and **87,585 catalog entries**. The rating record is extremely concentrated: among the **84,432 entries with a rating**, the most rated **1% receive 54.9%** of all ratings and the most rated **10% receive 95.7%**. These are observations about activity on MovieLens, whose interface and recommendations helped generate the record.

## How the data was collected

The [GroupLens README](https://files.grouplens.org/datasets/movielens/ml-32m-README.html) describes ratings and tags entered through MovieLens, a movie recommendation service. It says users were selected at random, each selected user had at least 20 ratings, catalog entries have at least one rating or tag, and demographic data are absent. [Harper and Konstan (2015)](https://files.grouplens.org/papers/harper-tiis2015.pdf) explain why the data depend on the site's design: displayed titles and predictions influence what gets rated, and many people enter old viewing histories in a short session. Random selection is a documented collection claim; the released tables alone cannot verify the sampling procedure.

All four CSVs match the README's MD5 hashes. The ratings, movies, links, and tags were converted once to typed Parquet; [the notebook](../notebooks/01_dataset_orientation.ipynb) calls reusable code in `src/`. IDs are 32-bit integers, ratings are 32-bit floats, and timestamps are 64-bit integers; all dates below use UTC. IMDb's [non-commercial datasets](https://www.imdb.com/interfaces/) were downloaded and joined by the linked IMDb ID as a separate, later snapshot. No TMDB key was set, so no TMDB API calls were made.

## What the data contains and what was verified

| Attribute | Meaning and observed coverage |
|---|---|
| `userId` | Anonymous MovieLens user key; 200,948 users appear in ratings. No age, location, gender, or other demographics are present. |
| `movieId` | MovieLens catalog key; 87,585 entries in `movies.csv` and `links.csv`. Every rated movie joins to the catalog. |
| `rating` | Explicit star score, 0.5–5.0 in half-step values; median 3.5. No out-of-range or off-step values. No half-star rating predates February 18, 2003. |
| `timestamp` | Seconds since the Unix epoch, describing entry time, not viewing time. Observed UTC range: 1995-01-09 11:46:44 to 2023-10-13 02:29:07. The README states the final date as October 12, 2023; its date appears to use a different timezone, which the documentation does not specify. |
| `title`, `genres` | Manually entered or imported title and pipe-separated MovieLens genre labels. Release year can be parsed from 86,968 titles after trimming whitespace; 617 fail, e.g. *The Millions Game (Das Millionenspiel)* has no trailing year and *Mona and the Time of Burning Love (1983))* has an extra parenthesis. Genres are multi-label: Drama occurs on 34,175 entries; 7,080 entries have `(no genres listed)`. |
| `tag` | User-entered phrase applied to a movie, not a controlled category. There are 2,000,072 tag applications by 15,848 users on 51,323 entries. |
| `imdbId`, `tmdbId` | Links to external title IDs. All 87,585 entries have an IMDb ID field; 124 lack a TMDB ID. |

Every included user has at least 20 ratings, confirming the floor. There are **0 repeated user–movie pairs**, **0 invalid ratings**, and **0 ratings whose movie ID is absent from the catalog**. There are **3,153 catalog entries without ratings**, all of which have tags, confirming the README's stated inclusion rule. The README's counts of users and movies match the files. Its first date matches UTC; its last calendar date differs from the UTC conversion by a few hours. The dataset cannot independently test the README's random-user claim.

User activity ranges from **20 to 33,332 ratings** (median **73**, 90th percentile **364**). **23,039 users, or 11.5%,** have 20–25 ratings, so the inclusion floor affects a visible portion of the sample. Movie activity has a median of **5** ratings among rated entries and a maximum of **102,929**, for *The Shawshank Redemption (1994)*. **40,548 rated entries (48.0%)** have fewer than 5 ratings; **61,082 (72.3%)** have fewer than 20; and **72,241 (85.6%)** have fewer than 100. A median-count example is *Hellhounds on My Trail (1999)* with 5; one-rating examples include *Condo Painting (2000)* and *Stacy's Knights (1982)*. The Gini coefficient of rating counts across rated entries is **0.952**. See Figures 01 and 02.

### Time, sessions, and platform history

The [documented launch](https://files.grouplens.org/papers/harper-tiis2015.pdf) was in August/fall 1997, but **2,159,894 ratings from 33,358 users** have timestamps before August 1, 1997; the count is unchanged using September 1. The leading pre-launch titles include *Batman (1989)* and *Dances with Wolves (1990)*. This is a material disagreement between the apparent timestamp interpretation and the platform timeline. **Hypotheses, not established facts:** migrated records, backdated entries, or a timestamp conversion/export problem. Harper and Konstan explicitly say the EachMovie seed ratings were not included in earlier MovieLens releases, so attributing these records to EachMovie would be premature. The proposal excludes this period from its main comparisons pending investigation.

Half-star values appear only after the documented v3 rating-scale change; their monthly share varies considerably later, so the scale change did not produce a clean, permanent level shift. **1,014 ratings on 100 titles** precede the year parsed from the title. Some may reflect advance screenings or title metadata errors; no explanation is established by these files. Figure 03 shows half-star use over time.

With a session defined by gaps under 30 minutes between one user's consecutive ratings, the median session has **1 rating**, but **21,347,597 ratings (66.7%)** occur in sessions of **50 or more**. The median gap between adjacent ratings is **10 seconds**, and **11,898,883 gaps** are at most **5 seconds**. The median first session contains **47 ratings**, consistent with bulk entry and the 20-rating sampling floor. A 60-minute gap rule raises the large-session share only to **67.7%** and the median first-session size to **48**. These statistics support caution about treating rating timestamps as viewing events. Figure 08 presents the sensitivity check.

Across all users' first sessions under the 30-minute rule, **16,166,278 ratings** were entered. **23.9%** of these first-session ratings went to films in the prior-year top 1% and **64.6%** to the full-export top 1%; the 60-minute rule gives **23.8%** and **64.3%** respectively. This checks first-session composition separately from the first-20 measure.

### First ratings, mainstreamness, and catalog growth

The main cohort comparison excludes users whose first rating predates January 1998 because the prior-year popularity baseline is too sparse and the pre-launch records are unresolved. This leaves **35,717 pre-v3**, **62,900 v3**, and **67,933 v4+** users. Era boundaries are February 18, 2003 and November 1, 2014 in UTC; the latter uses the documented month because no exact v4 day is given.

For each user's first 20 ratings, a movie is “already popular” if its count at the end of the **preceding calendar year** placed it in the top 1% of movies with any prior ratings. Movies with no prior ratings count as not already popular. Under this no-look-ahead measure, the share is **8.58% pre-v3**, **8.58% v3**, and **66.05% v4+**. In contrast, defining popularity using all 32 million future and past ratings yields **63.04%**, **63.52%**, and **85.02%**. This sensitivity is central: a full-data popularity measure substantially changes the story. Prior-history coverage is also uneven: **16.9%** of the stable pre-v3 cohort's first 20 films have no prior MovieLens rating, versus **1.9%** in v3 and **2.4%** in v4+. Figures 04 and 05 show the cohort pattern and the two definitions. These differences do not by themselves identify an interface effect because catalog size, cohort composition, and the ability to add movies also changed.

At monthly resolution, **267 October 2014 starters** direct **15.9%** of their first 20 ratings to the prior-year top 1%; **435 November 2014 starters** direct **57.1%**. This lines up with the documented v4 month, while the exact launch day and simultaneous changes are not separated. Figure 04 zooms in on this transition.

A user mainstreamness score takes the median prior-year popularity percentile of films they rated, restricted to users starting in 1998 or later with at least 10 ratings on films that already had ratings in a prior year and a prior-year catalog of at least 1,000 rated titles. **166,037 users** meet those rules; their median score is **0.963**, and the correlation with log rating count is **−0.074**. Scores are therefore only weakly associated with activity in this selected set. Era medians are **0.844**, **0.955**, and **0.989**, but shifts in catalog coverage can move percentiles. The six extreme users and their title/genre examples are reproducible in `eda_metrics.json`; even “niche” examples include globally familiar titles, illustrating that an on-site, time-relative score is not the same as world obscurity.

The first rating date is a proxy for catalog entry, not an official listing date. First-rated entries number **933 in 2008**, **1,910 in 2009**, and **7,798 in 2015**. These changes are consistent with a changing catalog and use of the site, but cannot be attributed to a single design change. Figure 06 shows the series.

Ratings average **3.54 stars** overall. Mean ratings by first-rating cohort are **3.54 pre-v3**, **3.49 v3**, and **3.58 v4+**; these are descriptive and combine different rating scales and user mixes. Movies in the highest decile by MovieLens rating count average **3.56 stars** across their ratings, compared with **2.87** for the lowest decile. **861 users** have a within-user rating standard deviation below 0.25 stars. Thus rating level and rating volume are related, while individual scale use varies.

### External popularity and tagging

The IMDb join matches a title record for **87,355 of 87,585** MovieLens entries (**99.7%**) and a vote count for **87,226** (**99.6%**); **84,108** entries have both a MovieLens rating count and an IMDb vote count. The Pearson correlation between `log(1 + count)` values is **0.850**. With top-decile cutoffs within these matched entries, **2,003** titles are popular on MovieLens but in the IMDb tail and **2,002** are in the opposite cell (Figure 07). For example, *Forget Paris (1995)* has 6,442 MovieLens ratings and 13,982 current IMDb votes. This is a descriptive cross-platform comparison, not a measure of IMDb popularity at the time of a MovieLens rating: IMDb's live dataset is refreshed daily and was retrieved in 2026, after the MovieLens endpoint. Some high-IMDb, low-MovieLens examples are 2023 releases and had little time to collect MovieLens ratings before the export.

IMDb classifies **10,058 matched MovieLens entries** as types other than `movie` or `tvMovie`, such as shorts and videos. This matters when defining the catalog analysis. IMDb's `title.akas` language and region columns describe localized alternative titles, not reliably a film's original language or production country. None of the matched records has a non-null language on a row marked as the original title, so the available IMDb files cannot defensibly quantify non-English or non-US production. A TMDB join or another source with production metadata is needed for that subquestion.

Tagging is highly concentrated: the most active **1% of taggers supply 65.6%** of tag applications, and one user applied **723,473** tags. Tagging is potentially informative but is secondary to the proposal's rating and catalog questions.

## What the data can and cannot support

The three post-1998 cohorts contain tens of thousands of users, enough for descriptive trends and tightly defined subgroup comparisons. The data can show when ratings were entered, which titles were rated, how this differs by cohort, and whether on-site popularity measures are sensitive to historical look-ahead. It cannot reveal which titles a user saw but chose not to rate, what recommendation or prediction was displayed, when a film was watched, why a user joined, or what would have happened under another interface. The 20-rating inclusion rule removes failed or brief onboarding experiences. The dataset has no demographics. Platform events often coincide with other changes and external film releases; thus “shaped” is a research question, not a causal conclusion from this dataset alone.

## Analyses proposed from the EDA

1. Use monthly first-rating cohorts around v3 and v4, display cohort size, and compare first-20 concentration with the no-look-ahead score. Repeat with a fixed set of films available before each event and with global popularity only as a sensitivity check.
2. Investigate pre-launch timestamps and exclude unresolved records from the primary time-series claims. Report the boundary sensitivity for August versus September 1997.
3. Analyze first-session concentration and catalog entry around September 2008 and November 2014. Repeat session measures with 30- and 60-minute gaps.
4. Compare MovieLens and IMDb popularity for titles old enough to have exposure in both snapshots; describe mismatches without treating today's IMDb votes as historical data.
5. Defer model amplification work until the descriptive platform analysis is stable. A later, small popularity baseline can establish a reference before any full recommender training.

## Sources

- [GroupLens MovieLens 32M README](https://files.grouplens.org/datasets/movielens/ml-32m-README.html).
- [F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*.](https://doi.org/10.1145/2827872)
- [Himan Abdollahpouri et al. 2019. *The Unfairness of Popularity Bias in Recommendation*.](https://arxiv.org/abs/1907.13286)
- [IMDb Non-Commercial Datasets and field definitions](https://www.imdb.com/interfaces/).
