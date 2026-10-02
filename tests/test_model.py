from pathlib import Path

import numpy as np
import pandas as pd

import model
from model import (
    DEFAULT_RATINGS_NROWS,
    MOVIES_PATH_ENV,
    PROJECT_DIR,
    RATINGS_MAX_ROWS_ENV,
    HybridMovieRecommender,
    load_data,
    parse_ratings_limit,
    resolve_data_path,
)


def test_item_knn_neighbor_uses_movie_id_title_mapping():
    # Catalog row order intentionally differs from sorted movieId/KNN order.
    movies = pd.DataFrame(
        [
            (30, "Movie 30", "Comedy"),
            (40, "Movie 40", "Drama"),
            (20, "Movie 20", "Action|Adventure"),
            (50, "Movie 50", "Romance"),
            (10, "Movie 10", "Action|Adventure"),
        ],
        columns=["movieId", "title", "genres"],
    )
    ratings = pd.DataFrame(
        [
            (1, 10, 5), (1, 20, 5), (1, 30, 1),
            (2, 10, 4), (2, 20, 4), (2, 30, 1),
            (3, 10, 5), (3, 20, 5), (3, 40, 3),
            (4, 30, 4), (4, 40, 4), (4, 50, 4),
        ],
        columns=["userId", "movieId", "rating"],
    )

    recommender = HybridMovieRecommender(movies, ratings)

    neighbor_id, _ = recommender._knn_neighbors(10, 1)[0]
    assert neighbor_id == 20
    assert recommender.movie_id_to_title[neighbor_id] == "Movie 20"
    assert recommender.collab_recommend("Movie 10", top_n=1) == ["Movie 20"]
    assert recommender.cold_start_recommend("Adven", top_n=5) == [
        "Movie 20",
        "Movie 10",
    ]


def test_collaborative_recommendations_keep_zero_cosine_neighbors():
    movies = pd.DataFrame(
        [(10, "Seed", "Action"), (20, "Orthogonal", "Drama")],
        columns=["movieId", "title", "genres"],
    )
    ratings = pd.DataFrame(
        [(1, 10, 5), (2, 20, 5)], columns=["userId", "movieId", "rating"]
    )

    recommender = HybridMovieRecommender(movies, ratings)

    assert recommender._knn_neighbors(10, 1) == [(20, 0.0)]
    assert recommender.collab_recommend("Seed", top_n=1) == ["Orthogonal"]


def test_hybrid_scores_keep_equal_content_and_collaborative_weights(monkeypatch):
    movies = pd.DataFrame(
        [
            (30, "Movie 30", "Comedy"),
            (40, "Movie 40", "Drama"),
            (20, "Movie 20", "Action"),
            (10, "Movie 10", "Action"),
        ],
        columns=["movieId", "title", "genres"],
    )
    ratings = pd.DataFrame(
        [(1, 10, 5), (1, 20, 4), (2, 10, 4), (2, 20, 5)],
        columns=["userId", "movieId", "rating"],
    )
    recommender = HybridMovieRecommender(movies, ratings)
    content = np.array([0.8, 0.3, 0.1, 1.0])
    monkeypatch.setattr(recommender, "_content_scores_for_movie", lambda _: content)
    monkeypatch.setattr(
        recommender,
        "_collaborative_scores_for_movie",
        lambda _movie_id, _top_n: {30: 0.1, 40: 0.9, 20: 0.5},
    )

    scores = recommender._hybrid_scores_for_movie(10, top_n=3)

    assert scores[recommender.movie_id_to_catalog_index[30]] == 0.45
    assert scores[recommender.movie_id_to_catalog_index[40]] == 0.6
    assert scores[recommender.movie_id_to_catalog_index[20]] == 0.3


def test_relative_data_paths_are_repository_relative_even_after_chdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(MOVIES_PATH_ENV, raising=False)

    assert resolve_data_path(
        None, env_var=MOVIES_PATH_ENV, default_name="movies.csv"
    ) == (PROJECT_DIR / "movies.csv").resolve()


def test_absolute_environment_data_path_is_respected(monkeypatch, tmp_path):
    path = tmp_path / "outside-ratings.csv"
    monkeypatch.setenv("RATINGS_CSV_PATH", str(path))

    assert resolve_data_path(
        None, env_var="RATINGS_CSV_PATH", default_name="ratings.csv"
    ) == path.resolve()


def test_relative_csv_paths_load_from_project_root_after_chdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    movies, ratings = load_data(
        "tests/fixtures/movies.csv", "tests/fixtures/ratings.csv"
    )

    assert len(movies) == 6
    assert len(ratings) == 16


def test_load_data_defaults_to_bounded_rows_and_allows_full_file(monkeypatch):
    real_read_csv = pd.read_csv
    rating_limits = []

    def record_read_csv(path, *args, **kwargs):
        if Path(path).name == "ratings.csv":
            rating_limits.append(kwargs.get("nrows"))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(model.pd, "read_csv", record_read_csv)
    movies_path = PROJECT_DIR / "tests" / "fixtures" / "movies.csv"
    ratings_path = PROJECT_DIR / "tests" / "fixtures" / "ratings.csv"

    load_data(movies_path, ratings_path)
    load_data(movies_path, ratings_path, ratings_nrows=None)

    assert rating_limits == [DEFAULT_RATINGS_NROWS, None]


def test_app_rating_limit_can_be_configured_by_environment(monkeypatch):
    movies = pd.read_csv(PROJECT_DIR / "tests" / "fixtures" / "movies.csv")
    ratings = pd.read_csv(PROJECT_DIR / "tests" / "fixtures" / "ratings.csv")
    selected_limits = []

    def fake_load_data(**kwargs):
        selected_limits.append(kwargs["ratings_nrows"])
        return movies, ratings

    monkeypatch.setattr(model, "load_data", fake_load_data)
    monkeypatch.setattr(model, "HybridMovieRecommender", lambda *_: "recommender")
    monkeypatch.delenv(RATINGS_MAX_ROWS_ENV, raising=False)
    assert model.load_default_recommender() == "recommender"

    monkeypatch.setenv(RATINGS_MAX_ROWS_ENV, "500000")
    assert model.load_default_recommender() == "recommender"

    monkeypatch.setenv(RATINGS_MAX_ROWS_ENV, "full")
    assert model.load_default_recommender() == "recommender"
    assert selected_limits == [DEFAULT_RATINGS_NROWS, 500000, None]
    assert parse_ratings_limit("all") is None
