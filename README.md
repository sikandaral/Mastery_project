# MovieLens platform-history capstone proposal

This repository contains the SI 699 MovieLens 32M dataset orientation, eight figures, and a three-page [proposal PDF](reports/project_proposal.pdf) for the October 8, 2026 deliverable. The [project outline](reports/project_outline.md) describes the intended later analysis. The raw datasets and processed Parquet files are intentionally not committed.

## Reproduce the analysis

Use Python 3.13 and install the pinned [requirements](requirements.txt). Download [MovieLens 32M](https://files.grouplens.org/datasets/movielens/ml-32m.zip), unzip to `data/raw/ml-32m/`, and download these [IMDb non-commercial files](https://www.imdb.com/interfaces/) to `data/raw/`:

- `title.basics.tsv.gz`
- `title.ratings.tsv.gz`
- `title.akas.tsv.gz`

Then run the [notebook](notebooks/01_dataset_orientation.ipynb) top to bottom from the repository root. It verifies the four MovieLens CSV MD5 hashes against the [GroupLens README](https://files.grouplens.org/datasets/movielens/ml-32m-README.html) before analysis, converts explicit typed columns to Parquet, computes the JSON metrics, and regenerates the PNG figures. Direct script equivalents are:

```bash
python src/eda.py
python src/external.py
python src/figures.py
```

The source files are in `src/`, computed values in `reports/eda_metrics.json` and `reports/external_metrics.json`, captions in `figures/captions.md`, and the narrative interpretation in [EDA findings](reports/eda_findings.md). The [key-number table](reports/key_numbers.md) maps proposal statistics to code. IMDb's files are updated daily; current vote counts will change if downloaded again. The project uses them as a later cross-platform snapshot, not as historical movie popularity.

The notebook performs EDA only. It does not train a recommender.

## Data attribution

MovieLens 32M is provided by GroupLens under its [dataset terms](https://files.grouplens.org/datasets/movielens/ml-32m-README.html). Cite F. Maxwell Harper and Joseph A. Konstan, “The MovieLens Datasets: History and Context,” *ACM Transactions on Interactive Intelligent Systems* 5(4), Article 19 (2015), [DOI: 10.1145/2827872](https://doi.org/10.1145/2827872). IMDb data come from its [non-commercial datasets](https://www.imdb.com/interfaces/).
