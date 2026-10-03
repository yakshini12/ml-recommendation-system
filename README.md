# Hybrid Movie Recommendation System

A small Flask application that returns content-based, item-based collaborative, and hybrid movie recommendations from a movie catalog and ratings CSV.

## Recommendation methods

- **Content-based:** TF-IDF vectors built from each movie's `genres` field, ranked by cosine similarity.
- **Collaborative:** item-based K-nearest neighbors over movie-by-user rating vectors, using cosine distance. Similarity scores are `1 - cosine distance`.
- **Hybrid:** adds content and collaborative similarity scores with equal weights (`0.5` each), then ranks the combined scores.
- **Genre popularity:** for a selected genre, ranks matching movies by the number of distinct user/movie interactions in the loaded data.

The title-based collaborative and hybrid methods recommend movies similar to a selected title. The per-user ranking method is used by the evaluation script.

## Data

`movies.csv` is included and must contain `movieId`, `title`, and `genres` columns. A `ratings.csv` file is needed to run the recommendation model and is not included in the repository. Provide a ratings file with `userId`, `movieId`, and `rating` columns. Its movie IDs must correspond to entries in the movie catalog for those titles to be recommendable.

Data paths do not depend on the directory from which a command is launched. By default, both CSV files are read from the repository root. Override them with `MOVIES_CSV_PATH` and `RATINGS_CSV_PATH`; each can be an absolute path or a path relative to the repository root. The Flask app reads at most the first 200,000 ratings rows by default. Set `RATINGS_MAX_ROWS` to a positive row count or `full` to change that limit. A root-level `ratings.csv` is ignored by Git so local data is not accidentally committed.

## Setup and run

Python 3.10 or newer is required. From the repository directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:RATINGS_CSV_PATH = "C:\path\to\ratings.csv"
python app.py
```

The landing page can load without ratings data. Submitting a recommendation requires a valid ratings CSV; the app displays a setup error if it is missing or malformed. The application listens on Flask's default local development address.

## Evaluation

Run the deterministic per-user holdout evaluation with:

```powershell
python evaluate.py --ratings C:\path\to\ratings.csv
```

Optional arguments are `--movies`, `--k`, `--test-fraction`, `--relevance-threshold`, and `--seed`. The defaults are `k=10`, a seeded per-user random holdout of 20% rounded up for each user, relevance threshold `4.0`, and seed `42`.

Evaluation reads the full ratings file by default, independently of the app's 200,000-row limit. Use `--ratings-rows N` to cap it, or `--ratings-rows full` to request all rows explicitly.

The split averages duplicate user/movie rows before splitting and keeps at least one training interaction for each user with two or more interactions. One item-based KNN is fitted on the combined training interactions and shared across users; holdout rows are not used to fit it. Relevant held-out movies are those rated at least the threshold and present in both the training item set and the catalog. The script reports macro Precision@K (hits divided by K for each evaluated user) and macro Recall@K (hits divided by that user's relevant held-out count), averaged over users with at least one eligible relevant holdout.

No dataset-wide Precision@10 or Recall@10 result is reported here because the repository does not include ratings data. The small CSVs in `tests/fixtures/` are synthetic test data, not benchmark results.

## Tests

Install the requirements, then run:

```powershell
python -m pytest
```
