# Hybrid Movie Recommendation System

A small Flask application that returns content-based, item-based collaborative, and hybrid movie recommendations from a movie catalog and ratings data.

## Recommendation methods

- **Content-based:** TF-IDF vectors built from each movie's `genres` field, ranked by cosine similarity.
- **Collaborative:** item-based K-nearest neighbors over movie-by-user rating vectors, using cosine distance. Similarity scores are `1 - cosine distance`.
- **Hybrid:** adds content and collaborative similarity scores with equal weights (`0.5` each), then ranks the combined scores.
- **Genre popularity:** for a selected genre, ranks matching movies by the number of distinct user/movie interactions in the loaded data.

The title-based collaborative and hybrid methods recommend movies similar to a selected title. The per-user ranking method is used by the evaluation script.

## Data

`movies.csv` is the tracked catalog and `data/demo_ratings.csv` is a small deterministic synthetic fixture bundled for the public demo. The Flask app uses a configured external ratings file when it exists; otherwise it falls back to the demo fixture. The public demo does not require or load the full MovieLens 25M ratings dataset. Demo ratings are for exercising the application, not for benchmark claims.

Data paths do not depend on the directory from which a command is launched. `MOVIES_CSV_PATH` and `RATINGS_CSV_PATH` accept absolute paths or paths relative to the repository root. The Flask app uses an external ratings file only when `RATINGS_CSV_PATH` points to an existing file; otherwise it loads the bundled demo fixture. External ratings are capped at 200,000 rows by default. Set `RATINGS_MAX_ROWS` to a positive row count or `full` to explicitly change that limit. Root-level and `data/ratings.csv` files are ignored by Git.

## Setup and run

Python 3.10 or newer is required. From the repository directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open the local address shown by Flask and enter `Toy Story (1995)` to see content, collaborative, and hybrid recommendations. The app also supports genre-based recommendations for a new user. No external ratings file is needed for this demo. To use an external file locally, set `RATINGS_CSV_PATH` to its path; if that path is absent, the app uses the bundled fixture. The default local host and port are `127.0.0.1:5000`; set `HOST` and `PORT` to override them.

## Deployment

The included `Procfile` starts the app with Gunicorn on the host-provided `PORT`. Install dependencies with `pip install -r requirements.txt`, then use this start command on a Procfile-compatible Python web host:

```sh
gunicorn --bind 0.0.0.0:$PORT app:app
```

The deployed demo uses the tracked `movies.csv` and bundled `data/demo_ratings.csv`; it does not need an external dataset.

## Evaluation

Run the deterministic per-user holdout evaluation with:

```powershell
python evaluate.py --movies movies.csv --ratings C:\path\to\ratings.csv --ratings-rows full
```

Optional arguments are `--movies`, `--k`, `--test-fraction`, `--relevance-threshold`, and `--seed`. The defaults are `k=10`, a seeded per-user random holdout of 20% rounded up for each user, relevance threshold `4.0`, and seed `42`.

Full MovieLens evaluation remains a separate local task and requires the external MovieLens files; those large files are not bundled with the app or repository. `evaluate.py` reads the full ratings file by default, independently of the app's 200,000-row limit. Use `--ratings-rows N` to cap it, or `--ratings-rows full` to request all rows explicitly.

The split averages duplicate user/movie rows before splitting and keeps at least one training interaction for each user with two or more interactions. One item-based KNN is fitted on the combined training interactions and shared across users; holdout rows are not used to fit it. Relevant held-out movies are those rated at least the threshold and present in both the training item set and the catalog. The script reports macro Precision@K (hits divided by K for each evaluated user) and macro Recall@K (hits divided by that user's relevant held-out count), averaged over users with at least one eligible relevant holdout.

No dataset-wide Precision@10 or Recall@10 result is reported here. The demo ratings and CSVs in `tests/fixtures/` are synthetic data, not benchmark results.

## Tests

Install the requirements, then run:

```powershell
python -m pytest
```
