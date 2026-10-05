"""Reproducible MovieLens 32M orientation. Run with Python 3.13 and requirements.txt."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))
import duckdb

RAW = ROOT / "data/raw/ml-32m"
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"


def connect():
    con = duckdb.connect(str(PROCESSED / "eda.duckdb"))
    con.execute("SET threads=4")
    con.execute("SET memory_limit='6GB'")
    con.execute("SET TimeZone='UTC'")
    con.execute("SET temp_directory='data/processed/tmp'")
    return con


def prepare(con):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "tmp").mkdir(exist_ok=True)
    queries = {
        "ratings": "SELECT CAST(userId AS INTEGER) userId, CAST(movieId AS INTEGER) movieId, CAST(rating AS FLOAT) rating, CAST(timestamp AS BIGINT) AS \"timestamp\" FROM read_csv_auto('data/raw/ml-32m/ratings.csv', header=true)",
        "movies": "SELECT CAST(movieId AS INTEGER) movieId, title, genres, TRY_CAST(regexp_extract(trim(title), '\\(([0-9]{4})\\)$', 1) AS INTEGER) release_year FROM read_csv_auto('data/raw/ml-32m/movies.csv', header=true)",
        "links": "SELECT CAST(movieId AS INTEGER) movieId, imdbId, tmdbId FROM read_csv_auto('data/raw/ml-32m/links.csv', header=true, all_varchar=true)",
        "tags": "SELECT CAST(userId AS INTEGER) userId, CAST(movieId AS INTEGER) movieId, tag, CAST(timestamp AS BIGINT) AS \"timestamp\" FROM read_csv_auto('data/raw/ml-32m/tags.csv', header=true)",
    }
    for name, query in queries.items():
        path = PROCESSED / f"{name}.parquet"
        if not path.exists() or name == "movies":
            con.execute(f"COPY ({query}) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)")
    for name in queries:
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('data/processed/{name}.parquet')")


def rows(con, sql):
    cur = con.execute(sql)
    return [dict(zip([x[0] for x in cur.description], row)) for row in cur.fetchall()]


def one(con, sql):
    return rows(con, sql)[0]


def analyze(con):
    out = {}
    out["sizes"] = one(con, """SELECT (SELECT count(*) FROM ratings) ratings,
      (SELECT count(DISTINCT userId) FROM ratings) users,
      (SELECT count(*) FROM movies) movies,
      (SELECT count(*) FROM tags) tags,
      (SELECT min(to_timestamp(timestamp)) FROM ratings) first_rating,
      (SELECT max(to_timestamp(timestamp)) FROM ratings) last_rating""")
    out["quality"] = one(con, """WITH pairs AS
      (SELECT count(*) n, count(DISTINCT (userId,movieId)) unique_pairs FROM ratings)
      SELECT n-unique_pairs duplicate_pairs,
       (SELECT count(*) FROM ratings WHERE rating < 0.5 OR rating > 5 OR rating*2 != floor(rating*2)) invalid_ratings,
       (SELECT count(*) FROM ratings r ANTI JOIN movies m USING (movieId)) ratings_missing_movie,
       (SELECT count(*) FROM movies m ANTI JOIN ratings r USING (movieId)) movies_without_rating,
       (SELECT count(*) FROM movies m ANTI JOIN ratings r USING (movieId) SEMI JOIN tags t USING (movieId)) tag_only_movies,
       (SELECT count(*) FROM movies WHERE release_year IS NULL) missing_title_year,
       (SELECT count(*) FROM movies WHERE genres='(no genres listed)') no_genres,
       (SELECT count(*) FROM links WHERE imdbId IS NULL OR imdbId='') no_imdb_id,
       (SELECT count(*) FROM links WHERE tmdbId IS NULL OR tmdbId='') no_tmdb_id
       FROM pairs""")
    out["year_failure_examples"] = rows(con, "SELECT movieId,title FROM movies WHERE release_year IS NULL ORDER BY movieId LIMIT 12")
    out["genres"] = rows(con, "SELECT genre, count(*) movies FROM (SELECT unnest(string_split(genres,'|')) genre FROM movies) GROUP BY 1 ORDER BY movies DESC")
    con.execute("CREATE OR REPLACE TEMP TABLE user_counts AS SELECT userId,count(*) n FROM ratings GROUP BY 1")
    con.execute("CREATE OR REPLACE TEMP TABLE movie_counts AS SELECT movieId,count(*) n FROM ratings GROUP BY 1")
    out["user_activity"] = one(con, """SELECT min(n) min, median(n) median, max(n) max,
      quantile_cont(n,.9) p90,quantile_cont(n,.99) p99,
      count(*) FILTER (WHERE n<=25) users_20_to_25,
      count(*) FILTER (WHERE n=20) users_at_20 FROM user_counts""")
    out["movie_activity"] = one(con, """SELECT min(n) min, median(n) median, max(n) max,
      count(*) FILTER (WHERE n<5) under_5, count(*) FILTER (WHERE n<20) under_20,
      count(*) FILTER (WHERE n<100) under_100 FROM movie_counts""")
    out["user_histogram"] = rows(con, "SELECT floor(log10(n)) log10_bin,count(*) users FROM user_counts GROUP BY 1 ORDER BY 1")
    out["movie_histogram"] = rows(con, "SELECT floor(log10(n)) log10_bin,count(*) movies FROM movie_counts GROUP BY 1 ORDER BY 1")
    out["concentration"] = one(con, """WITH ranked AS
      (SELECT n, row_number() OVER (ORDER BY n DESC) rank, count(*) OVER () movies,
       sum(n) OVER () ratings, sum(n) OVER (ORDER BY n DESC ROWS UNBOUNDED PRECEDING) cum_n FROM movie_counts),
      g AS (SELECT sum(rank*n) weighted, sum(n) total, count(*) movies FROM
          (SELECT n,row_number() OVER (ORDER BY n) rank FROM movie_counts))
      SELECT max(cum_n / ratings) FILTER (WHERE rank=ceil(ranked.movies*.01)) top_1_share,
       max(cum_n / ratings) FILTER (WHERE rank=ceil(ranked.movies*.10)) top_10_share,
       (2.0*max(weighted))/(max(g.movies)*max(total))-(max(g.movies)+1.0)/max(g.movies) gini FROM ranked CROSS JOIN g""")
    out["movie_examples"] = rows(con, """WITH candidates AS
      (SELECT c.movieId,m.title,c.n,row_number() OVER (PARTITION BY c.n ORDER BY c.movieId) rn
       FROM movie_counts c JOIN movies m USING(movieId)
       WHERE c.n=(SELECT max(n) FROM movie_counts) OR c.n=1 OR
       c.n=(SELECT median(n) FROM movie_counts))
      SELECT movieId,title,n FROM candidates WHERE rn<=3 ORDER BY n DESC,movieId""")
    out["rating_scale"] = one(con, """SELECT min(rating) min, median(rating) median,max(rating) max,
      avg(rating) mean, count(*) FILTER (WHERE rating*2 != floor(rating*2)) invalid_half_step,
      count(*) FILTER (WHERE rating != floor(rating) AND timestamp < epoch(TIMESTAMP '2003-02-18')) pre_v3_half_stars,
      count(*) FILTER (WHERE timestamp < epoch(TIMESTAMP '1997-09-01')) pre_sep_1997_ratings,
      count(DISTINCT userId) FILTER (WHERE timestamp < epoch(TIMESTAMP '1997-09-01')) pre_sep_1997_users,
      count(*) FILTER (WHERE timestamp < epoch(TIMESTAMP '1997-08-01')) pre_aug_1997_ratings,
      count(DISTINCT userId) FILTER (WHERE timestamp < epoch(TIMESTAMP '1997-08-01')) pre_aug_1997_users
      FROM ratings""")
    out["rating_distribution"] = rows(con,"SELECT rating,count(*) n FROM ratings GROUP BY 1 ORDER BY 1")
    out["rating_by_movie_popularity"] = rows(con,"""WITH ranked AS
      (SELECT movieId,n,ntile(10) OVER (ORDER BY n,movieId) decile FROM movie_counts)
      SELECT decile,count(*) ratings,avg(r.rating) mean_rating,median(m.n) median_movie_count
      FROM ratings r JOIN ranked m USING(movieId) GROUP BY 1 ORDER BY 1""")
    out["pre_launch_movies"] = rows(con, """SELECT r.movieId,m.title,count(*) n FROM ratings r JOIN movies m USING(movieId)
      WHERE timestamp < epoch(TIMESTAMP '1997-09-01') GROUP BY 1,2 ORDER BY n DESC LIMIT 10""")
    out["pre_launch_user_examples"] = rows(con,"""SELECT userId,count(*) n,min(to_timestamp(timestamp)) first_date,
      max(to_timestamp(timestamp)) last_date FROM ratings WHERE timestamp<epoch(TIMESTAMP '1997-09-01')
      GROUP BY 1 ORDER BY n DESC LIMIT 5""")
    out["pre_release"] = one(con, """SELECT count(*) n, count(DISTINCT r.movieId) movies FROM ratings r JOIN movies m USING(movieId)
      WHERE m.release_year IS NOT NULL AND year(to_timestamp(r.timestamp)) < m.release_year""")
    out["pre_release_examples"] = rows(con, """SELECT m.title,m.release_year,year(to_timestamp(r.timestamp)) rating_year,count(*) n
       FROM ratings r JOIN movies m USING(movieId) WHERE m.release_year IS NOT NULL
       AND year(to_timestamp(r.timestamp)) < m.release_year GROUP BY 1,2,3 ORDER BY n DESC LIMIT 8""")
    out["monthly"] = rows(con, """SELECT date_trunc('month',to_timestamp(timestamp)) month_date,count(*) ratings,
      avg(CASE WHEN rating != floor(rating) THEN 1 ELSE 0 END) half_share
      FROM ratings GROUP BY 1 ORDER BY 1""")
    con.execute("""CREATE OR REPLACE TEMP TABLE user_start AS SELECT userId,min(timestamp) first_ts,
      CASE WHEN min(timestamp)<epoch(TIMESTAMP '2003-02-18') THEN 'pre-v3'
           WHEN min(timestamp)<epoch(TIMESTAMP '2014-11-01') THEN 'v3' ELSE 'v4+' END era
      FROM ratings GROUP BY 1""")
    out["cohorts"] = rows(con, """SELECT era,count(*) users, median(u.n) median_total_ratings,
      min(to_timestamp(first_ts)) first_date, max(to_timestamp(first_ts)) last_date
      FROM user_start s JOIN user_counts u USING(userId) GROUP BY 1 ORDER BY first_date""")
    out["new_users_monthly"] = rows(con, "SELECT date_trunc('month',to_timestamp(first_ts)) month_date,count(*) users FROM user_start GROUP BY 1 ORDER BY 1")
    out["cohorts_yearly"] = rows(con, "SELECT year(to_timestamp(first_ts)) cohort_year,count(*) users FROM user_start GROUP BY 1 ORDER BY 1")
    out["rating_by_era"] = rows(con, """SELECT s.era,count(*) ratings,avg(r.rating) mean_rating,
      avg(CASE WHEN r.rating != floor(r.rating) THEN 1 ELSE 0 END) half_share
      FROM ratings r JOIN user_start s USING(userId) GROUP BY 1 ORDER BY 1""")
    out["user_rating_variance"] = one(con, """SELECT count(*) FILTER (WHERE sd<.25) nearly_constant,
      median(mean_rating) median_user_mean FROM
      (SELECT userId,avg(rating) mean_rating,stddev_pop(rating) sd FROM ratings GROUP BY 1)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE first20 AS SELECT userId,movieId,rating,timestamp,
       row_number() OVER (PARTITION BY userId ORDER BY timestamp,movieId) ord
       FROM ratings QUALIFY ord<=20""")
    con.execute("""CREATE OR REPLACE TEMP TABLE global_pop AS SELECT movieId,n,
      row_number() OVER (ORDER BY n DESC,movieId) rank FROM movie_counts""")
    out["first20_global"] = rows(con, """SELECT s.era,count(DISTINCT f.userId) users,
      avg(log(1+g.n)) mean_log_count,
      avg(CASE WHEN g.rank<=ceil((SELECT count(*) FROM movie_counts)*.01) THEN 1 ELSE 0 END) top1_share
      FROM first20 f JOIN user_start s USING(userId) JOIN global_pop g USING(movieId) GROUP BY 1 ORDER BY 1""")
    out["first20_global_stable"] = rows(con, """SELECT s.era,count(DISTINCT f.userId) users,
      avg(log(1+g.n)) mean_log_count,
      avg(CASE WHEN g.rank<=ceil((SELECT count(*) FROM movie_counts)*.01) THEN 1 ELSE 0 END) top1_share
      FROM first20 f JOIN user_start s USING(userId) JOIN global_pop g USING(movieId)
      WHERE s.first_ts>=epoch(TIMESTAMP '1998-01-01') GROUP BY 1 ORDER BY 1""")
    con.execute("""CREATE OR REPLACE TEMP TABLE movie_year AS SELECT movieId,
      year(to_timestamp(timestamp)) AS \"year\",count(*) n FROM ratings GROUP BY 1,2""")
    con.execute("""CREATE OR REPLACE TEMP TABLE prior_pop AS SELECT m.movieId,y.year,
      coalesce(sum(my.n) FILTER (WHERE my.year<y.year),0) prior_n
      FROM movies m CROSS JOIN (SELECT DISTINCT year FROM movie_year) y
      LEFT JOIN movie_year my ON my.movieId=m.movieId AND my.year<y.year
      GROUP BY 1,2""")
    con.execute("""CREATE OR REPLACE TEMP TABLE prior_rank AS SELECT movieId,year,prior_n,
      row_number() OVER (PARTITION BY year ORDER BY prior_n DESC,movieId) rank,
      count(*) FILTER (WHERE prior_n>0) OVER (PARTITION BY year) active_movies
      FROM prior_pop""")
    out["first20_prior"] = rows(con, """SELECT s.era,count(DISTINCT f.userId) users,
      avg(log(1+p.prior_n)) mean_log_prior_count,
      avg(CASE WHEN p.prior_n>0 AND p.rank<=ceil(p.active_movies*.01) THEN 1 ELSE 0 END) prior_top1_share,
      avg(CASE WHEN p.prior_n=0 THEN 1 ELSE 0 END) no_prior_ratings_share
      FROM first20 f JOIN user_start s USING(userId)
      JOIN prior_rank p ON p.movieId=f.movieId AND p.year=year(to_timestamp(f.timestamp))
      GROUP BY 1 ORDER BY 1""")
    out["first20_prior_stable"] = rows(con, """SELECT s.era,count(DISTINCT f.userId) users,
      avg(log(1+p.prior_n)) mean_log_prior_count,
      avg(CASE WHEN p.prior_n>0 AND p.rank<=ceil(p.active_movies*.01) THEN 1 ELSE 0 END) prior_top1_share,
      avg(CASE WHEN p.prior_n=0 THEN 1 ELSE 0 END) no_prior_ratings_share
      FROM first20 f JOIN user_start s USING(userId)
      JOIN prior_rank p ON p.movieId=f.movieId AND p.year=year(to_timestamp(f.timestamp))
      WHERE s.first_ts>=epoch(TIMESTAMP '1998-01-01') GROUP BY 1 ORDER BY 1""")
    out["first20_yearly"] = rows(con, """SELECT year(to_timestamp(s.first_ts)) cohort_year,
      count(DISTINCT f.userId) users,avg(log(1+p.prior_n)) mean_log_prior_count,
      avg(CASE WHEN p.prior_n>0 AND p.rank<=ceil(p.active_movies*.01) THEN 1 ELSE 0 END) prior_top1_share
      FROM first20 f JOIN user_start s USING(userId)
      JOIN prior_rank p ON p.movieId=f.movieId AND p.year=year(to_timestamp(f.timestamp))
      GROUP BY 1 ORDER BY 1""")
    con.execute("""CREATE OR REPLACE TEMP TABLE user_mainstream AS SELECT r.userId,
      median(1.0-p.rank*1.0/p.active_movies) mainstream,
      count(*) n FROM ratings r JOIN prior_rank p ON p.movieId=r.movieId
      AND p.year=year(to_timestamp(r.timestamp)) JOIN user_start s ON s.userId=r.userId
      WHERE p.prior_n>0 AND p.active_movies>=1000 AND s.first_ts>=epoch(TIMESTAMP '1998-01-01')
      GROUP BY 1 HAVING count(*)>=10""")
    out["mainstream"] = one(con, "SELECT count(*) users,min(mainstream) min,median(mainstream) median,max(mainstream) max,quantile_cont(mainstream,.1) p10,quantile_cont(mainstream,.9) p90,corr(mainstream,log(1+n)) activity_corr FROM user_mainstream")
    out["mainstream_by_era"] = rows(con, "SELECT s.era,count(*) users,median(u.mainstream) median_mainstream FROM user_mainstream u JOIN user_start s USING(userId) GROUP BY 1 ORDER BY 1")
    out["extreme_users"] = rows(con, """WITH ranked AS (SELECT userId,mainstream,n,row_number() OVER (ORDER BY mainstream,userId) low_rank,
      row_number() OVER (ORDER BY mainstream DESC,userId) high_rank FROM user_mainstream)
      SELECT * FROM ranked WHERE low_rank<=3 OR high_rank<=3 ORDER BY mainstream""")
    out["extreme_user_titles"] = rows(con,"""WITH ranked AS (SELECT userId,mainstream,
      row_number() OVER (ORDER BY mainstream,userId) low_rank,
      row_number() OVER (ORDER BY mainstream DESC,userId) high_rank FROM user_mainstream),
      chosen AS (SELECT * FROM ranked WHERE low_rank<=3 OR high_rank<=3),
      rated AS (SELECT r.userId,m.title,m.genres,rank() OVER (PARTITION BY r.userId
       ORDER BY g.n DESC,r.movieId) rn FROM ratings r JOIN chosen c USING(userId)
       JOIN movies m USING(movieId) JOIN global_pop g USING(movieId))
      SELECT userId,title,genres FROM rated WHERE rn<=5 ORDER BY userId,rn""")
    out["catalog"] = rows(con, """WITH entry AS (SELECT m.movieId,m.release_year,
      year(to_timestamp(min(r.timestamp))) entry_year FROM ratings r JOIN movies m USING(movieId) GROUP BY 1,2)
      SELECT entry_year,count(*) entered,median(entry_year-release_year) median_lag,
      count(*) FILTER (WHERE entry_year<release_year) negative_lag FROM entry
      GROUP BY 1 ORDER BY 1""")
    out["tags"] = one(con, """SELECT count(DISTINCT userId) users,count(DISTINCT movieId) movies,
      (SELECT max(n) FROM (SELECT count(*) n FROM tags GROUP BY userId)) max_tags_user FROM tags""")
    out["tag_concentration"] = one(con, """WITH x AS (SELECT userId,count(*) n FROM tags GROUP BY 1),r AS
      (SELECT n,row_number() OVER (ORDER BY n DESC) rank,count(*) OVER() users,sum(n) OVER() tags,
       sum(n) OVER (ORDER BY n DESC ROWS UNBOUNDED PRECEDING) cum FROM x)
      SELECT max(cum/tags) FILTER (WHERE rank=ceil(users*.01)) top1_users_tag_share FROM r""")
    out["links"] = one(con, "SELECT count(*) links,count(DISTINCT movieId) linked_movies FROM links")
    return out


def sessions(con):
    out = {}
    con.execute("""CREATE OR REPLACE TEMP TABLE ordered AS SELECT userId,movieId,rating,timestamp,
      timestamp-lag(timestamp) OVER (PARTITION BY userId ORDER BY timestamp,movieId) gap
      FROM ratings""")
    out["gaps"] = one(con, """SELECT median(gap) median_seconds,
      count(*) FILTER (WHERE gap<=5) within_5_seconds,
      count(*) FILTER (WHERE gap<=30) within_30_seconds,
      count(*) FILTER (WHERE gap IS NOT NULL) total_gaps FROM ordered""")
    for gap_minutes in (30,60):
        con.execute(f"""CREATE OR REPLACE TEMP TABLE session_rows AS SELECT *,
          sum(CASE WHEN gap IS NULL OR gap>={gap_minutes*60} THEN 1 ELSE 0 END)
          OVER (PARTITION BY userId ORDER BY timestamp,movieId ROWS UNBOUNDED PRECEDING) sid FROM ordered""")
        con.execute("""CREATE OR REPLACE TEMP TABLE session_sizes AS SELECT userId,sid,count(*) n,
          min(timestamp) start_ts FROM session_rows GROUP BY 1,2""")
        out[f"sessions_{gap_minutes}"] = one(con, """SELECT count(*) sessions,median(n) median_size,max(n) max_size,
          sum(n) FILTER (WHERE n>=50) ratings_in_50plus,
          count(*) FILTER (WHERE n>=50) sessions_50plus FROM session_sizes""")
        out[f"first_session_{gap_minutes}"] = one(con, """WITH x AS (SELECT *,row_number() OVER
          (PARTITION BY userId ORDER BY start_ts,sid) rn FROM session_sizes)
          SELECT median(n) median_size,avg(n) mean_size,
          count(*) FILTER (WHERE n>=20) first_session_20plus FROM x WHERE rn=1""")
    return out


def main():
    os.chdir(ROOT)
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    con = connect()
    prepare(con)
    out = analyze(con)
    out.update(sessions(con))
    (REPORTS / "eda_metrics.json").write_text(json.dumps(out,indent=2,default=str))
    con.close()
    return out


if __name__ == "__main__":
    main()
