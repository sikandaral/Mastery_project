"""Join the current IMDb non-commercial snapshot to MovieLens IDs.

Run after src/eda.py. IMDb vote counts are current, not historical exposures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.eda import PROCESSED, REPORTS, connect, prepare, rows, one


def main():
    con = connect()
    prepare(con)
    con.execute("""CREATE OR REPLACE TEMP VIEW imdb_links AS SELECT movieId,
      CASE WHEN imdbId IS NOT NULL AND regexp_full_match(imdbId,'[0-9]+')
       THEN 'tt'||lpad(imdbId,greatest(7,length(imdbId))::INTEGER,'0') ELSE NULL END tconst
      FROM links""")
    files = {
        "basics": "data/raw/title.basics.tsv.gz",
        "ratings": "data/raw/title.ratings.tsv.gz",
        "akas": "data/raw/title.akas.tsv.gz",
    }
    for key,path in files.items():
        if not (ROOT/path).exists():
            raise FileNotFoundError(path)
    con.execute("""COPY (SELECT b.tconst,b.titleType,b.primaryTitle,b.originalTitle,
      try_cast(b.startYear AS INTEGER) startYear
      FROM read_csv('data/raw/title.basics.tsv.gz',delim='\t',header=true,all_varchar=true,nullstr='\\N',quote='') b
      SEMI JOIN imdb_links l USING(tconst))
      TO 'data/processed/imdb_basics_matched.parquet' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    con.execute("""COPY (SELECT r.tconst,try_cast(r.numVotes AS INTEGER) numVotes,
      try_cast(r.averageRating AS FLOAT) averageRating
      FROM read_csv('data/raw/title.ratings.tsv.gz',delim='\t',header=true,all_varchar=true,nullstr='\\N',quote='') r
      SEMI JOIN imdb_links l USING(tconst))
      TO 'data/processed/imdb_ratings_matched.parquet' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    con.execute("""COPY (SELECT a.titleId,a.region,a.language,a.isOriginalTitle
      FROM read_csv('data/raw/title.akas.tsv.gz',delim='\t',header=true,all_varchar=true,nullstr='\\N',quote='') a
      SEMI JOIN imdb_links l ON a.titleId=l.tconst)
      TO 'data/processed/imdb_akas_matched.parquet' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    con.execute("CREATE OR REPLACE TEMP VIEW imdb_b AS SELECT * FROM read_parquet('data/processed/imdb_basics_matched.parquet')")
    con.execute("CREATE OR REPLACE TEMP VIEW imdb_r AS SELECT * FROM read_parquet('data/processed/imdb_ratings_matched.parquet')")
    con.execute("CREATE OR REPLACE TEMP VIEW imdb_a AS SELECT * FROM read_parquet('data/processed/imdb_akas_matched.parquet')")
    con.execute("""CREATE OR REPLACE TEMP VIEW joined AS SELECT m.movieId,m.title,l.tconst,
       b.titleType,b.startYear,r.numVotes,c.n ml_ratings
       FROM movies m LEFT JOIN imdb_links l USING(movieId)
       LEFT JOIN imdb_b b USING(tconst) LEFT JOIN imdb_r r USING(tconst)
       LEFT JOIN (SELECT movieId,count(*) n FROM ratings GROUP BY 1) c USING(movieId)""")
    out = {}
    out['match'] = one(con, """SELECT count(*) movies,
       count(tconst) with_id,count(titleType) basics_matched,count(numVotes) votes_matched,
       count(*) FILTER (WHERE titleType IS NOT NULL AND titleType NOT IN ('movie','tvMovie')) non_movie_types,
       count(*) FILTER (WHERE ml_ratings IS NOT NULL AND numVotes IS NOT NULL) dual_popularity
       FROM joined""")
    out['correlation'] = one(con, """SELECT corr(log(1+ml_ratings),log(1+numVotes)) pearson_log,
       corr(ml_ratings,numVotes) pearson_raw FROM joined WHERE ml_ratings IS NOT NULL AND numVotes IS NOT NULL""")
    out['unmatched_examples'] = rows(con, """SELECT movieId,title,tconst,titleType,numVotes,ml_ratings
       FROM joined WHERE tconst IS NULL OR titleType IS NULL OR numVotes IS NULL
       ORDER BY ml_ratings DESC NULLS LAST LIMIT 15""")
    out['imdb_type_counts'] = rows(con,"SELECT coalesce(titleType,'unmatched') titleType,count(*) movies FROM joined GROUP BY 1 ORDER BY movies DESC")
    out['akas_coverage'] = one(con, """SELECT count(DISTINCT titleId) titles_with_akas,
       count(DISTINCT titleId) FILTER (WHERE language IS NOT NULL) titles_with_aka_language,
       count(DISTINCT titleId) FILTER (WHERE region IS NOT NULL) titles_with_aka_region,
       count(DISTINCT titleId) FILTER (WHERE isOriginalTitle='1' AND language IS NOT NULL) originals_with_language
       FROM imdb_a""")
    out['quadrants'] = rows(con, """WITH cut AS
       (SELECT quantile_cont(ml_ratings,.9) ml_cut,quantile_cont(numVotes,.9) imdb_cut
        FROM joined WHERE ml_ratings IS NOT NULL AND numVotes IS NOT NULL),
       q AS (SELECT j.*, CASE WHEN ml_ratings>=ml_cut THEN 'ML popular' ELSE 'ML tail' END ml_group,
        CASE WHEN numVotes>=imdb_cut THEN 'IMDb popular' ELSE 'IMDb tail' END imdb_group
        FROM joined j CROSS JOIN cut WHERE ml_ratings IS NOT NULL AND numVotes IS NOT NULL)
       SELECT ml_group,imdb_group,count(*) movies,median(ml_ratings) median_ml,
         median(numVotes) median_imdb FROM q GROUP BY 1,2 ORDER BY 1,2""")
    out['quadrant_examples'] = rows(con, """WITH cut AS
       (SELECT quantile_cont(ml_ratings,.9) ml_cut,quantile_cont(numVotes,.9) imdb_cut
        FROM joined WHERE ml_ratings IS NOT NULL AND numVotes IS NOT NULL),
       q AS (SELECT j.*, CASE WHEN ml_ratings>=ml_cut THEN 'ML popular' ELSE 'ML tail' END ml_group,
        CASE WHEN numVotes>=imdb_cut THEN 'IMDb popular' ELSE 'IMDb tail' END imdb_group,
        row_number() OVER (PARTITION BY (ml_ratings>=ml_cut),(numVotes>=imdb_cut)
          ORDER BY greatest(ml_ratings,1)*greatest(numVotes,1) DESC,movieId) rn
        FROM joined j CROSS JOIN cut WHERE ml_ratings IS NOT NULL AND numVotes IS NOT NULL)
       SELECT ml_group,imdb_group,movieId,title,ml_ratings,numVotes FROM q WHERE rn<=3 ORDER BY 1,2,rn""")
    (REPORTS/'external_metrics.json').write_text(json.dumps(out,indent=2,default=str))
    con.close()
    return out


if __name__=='__main__':
    main()
