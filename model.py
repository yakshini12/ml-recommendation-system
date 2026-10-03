"""Movie recommendation models and repository-relative data loading."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


PROJECT_DIR = Path(__file__).resolve().parent
MOVIES_PATH_ENV = "MOVIES_CSV_PATH"
RATINGS_PATH_ENV = "RATINGS_CSV_PATH"
RATINGS_MAX_ROWS_ENV = "RATINGS_MAX_ROWS"
DEFAULT_RATINGS_NROWS = 200_000
DEMO_RATINGS_PATH = PROJECT_DIR / "data" / "demo_ratings.csv"


def parse_ratings_limit(value: str) -> int | None:
    """Parse a positive row limit or ``full``/``all`` for the entire file."""
    normalized = str(value).strip().casefold()
    if normalized in {"full", "all"}:
        return None
    try:
        limit = int(normalized)
    except ValueError as error:
        raise ValueError("Ratings row limit must be a positive integer or 'full'.") from error
    if limit <= 0:
        raise ValueError("Ratings row limit must be greater than zero or 'full'.")
    return limit


def resolve_data_path(
    value: str | Path | None,
    *,
    env_var: str,
    default_name: str,
) -> Path:
    """Resolve a data path independently of the caller's current directory.

    Explicit arguments take precedence over environment variables. Relative
    paths are relative to the repository root; absolute paths can point to
    datasets stored elsewhere on the machine.
    """
    raw_path = value if value is not None else os.environ.get(env_var, default_name)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def load_data(
    movies_path: str | Path | None = None,
    ratings_path: str | Path | None = None,
    *,
    ratings_nrows: int | None = DEFAULT_RATINGS_NROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSVs; ratings default to 200,000 rows, or all rows when None."""
    if ratings_nrows is not None and ratings_nrows <= 0:
        raise ValueError("ratings_nrows must be greater than zero or None.")
    resolved_movies = resolve_data_path(
        movies_path, env_var=MOVIES_PATH_ENV, default_name="movies.csv"
    )
    resolved_ratings = resolve_data_path(
        ratings_path, env_var=RATINGS_PATH_ENV, default_name="ratings.csv"
    )

    if not resolved_movies.is_file():
        raise FileNotFoundError(
            f"Movie catalog not found at {resolved_movies}. "
            f"Set {MOVIES_PATH_ENV} to its CSV path."
        )
    if not resolved_ratings.is_file():
        raise FileNotFoundError(
            f"Ratings data not found at {resolved_ratings}. Add a ratings.csv "
            f"file or set {RATINGS_PATH_ENV} to its CSV path."
        )

    return pd.read_csv(resolved_movies), pd.read_csv(
        resolved_ratings, nrows=ratings_nrows
    )


def normalize_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Validate ratings and average duplicate user/movie interactions."""
    required = {"userId", "movieId", "rating"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"Ratings data is missing required columns: {sorted(missing)}")

    cleaned = ratings.loc[:, ["userId", "movieId", "rating"]].copy()
    for column in ("userId", "movieId", "rating"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
    cleaned = cleaned.dropna(subset=["userId", "movieId", "rating"])
    if cleaned.empty:
        raise ValueError("Ratings data must contain at least one valid interaction.")

    for column in ("userId", "movieId"):
        if not np.equal(np.mod(cleaned[column].to_numpy(), 1), 0).all():
            raise ValueError(f"Ratings column {column!r} must contain integer IDs.")
        cleaned[column] = cleaned[column].astype("int64")

    if not np.isfinite(cleaned["rating"].to_numpy(dtype=float)).all():
        raise ValueError("Ratings values must be finite numbers.")

    return (
        cleaned.groupby(["userId", "movieId"], as_index=False, sort=True)["rating"]
        .mean()
        .reset_index(drop=True)
    )


class HybridMovieRecommender:
    """Content, item-based collaborative, and 50/50 hybrid recommendations."""

    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
        required_movies = {"movieId", "title", "genres"}
        missing_movies = required_movies.difference(movies.columns)
        if missing_movies:
            raise ValueError(
                f"Movie data is missing required columns: {sorted(missing_movies)}"
            )

        catalog = movies.loc[:, ["movieId", "title", "genres"]].copy()
        catalog["movieId"] = pd.to_numeric(catalog["movieId"], errors="raise")
        if not np.equal(np.mod(catalog["movieId"].to_numpy(), 1), 0).all():
            raise ValueError("Movie catalog movieId values must be integers.")
        catalog["movieId"] = catalog["movieId"].astype("int64")
        catalog["title"] = catalog["title"].fillna("").astype(str).str.strip()
        catalog["genres"] = catalog["genres"].fillna("").astype(str)
        catalog = catalog.drop_duplicates("movieId", keep="first").reset_index(drop=True)
        if catalog.empty:
            raise ValueError("Movie catalog must contain at least one movie.")

        self.movies = catalog
        self.ratings = normalize_ratings(ratings)
        self.movie_id_to_catalog_index = {
            int(movie_id): index
            for index, movie_id in enumerate(self.movies["movieId"].tolist())
        }
        self.movie_id_to_title = {
            int(row.movieId): row.title for row in self.movies.itertuples(index=False)
        }
        self.title_to_movie_id: dict[str, int] = {}
        for row in self.movies.itertuples(index=False):
            self.title_to_movie_id.setdefault(row.title.casefold(), int(row.movieId))

        self._vectorizer = TfidfVectorizer(stop_words="english")
        try:
            self._genre_vectors = self._vectorizer.fit_transform(self.movies["genres"])
        except ValueError:
            # An all-empty genre column is valid input; it simply has no
            # content signal, while collaborative recommendations still work.
            self._genre_vectors = csr_matrix((len(self.movies), 0), dtype=float)

        self.item_movie_ids = np.asarray(
            sorted(int(movie_id) for movie_id in self.ratings["movieId"].unique()),
            dtype=np.int64,
        )
        self.user_ids = np.asarray(
            sorted(int(user_id) for user_id in self.ratings["userId"].unique()),
            dtype=np.int64,
        )
        self.movie_id_to_knn_index = {
            int(movie_id): index for index, movie_id in enumerate(self.item_movie_ids)
        }
        user_id_to_index = {int(user_id): index for index, user_id in enumerate(self.user_ids)}
        movie_id_to_index = self.movie_id_to_knn_index
        item_rows = self.ratings["movieId"].map(movie_id_to_index).to_numpy(dtype=np.int64)
        user_columns = self.ratings["userId"].map(user_id_to_index).to_numpy(dtype=np.int64)
        values = self.ratings["rating"].to_numpy(dtype=float)
        self.item_user_matrix = csr_matrix(
            (values, (item_rows, user_columns)),
            shape=(len(self.item_movie_ids), len(self.user_ids)),
            dtype=float,
        )

        self.user_ratings: dict[int, dict[int, float]] = {}
        for row in self.ratings.itertuples(index=False):
            self.user_ratings.setdefault(int(row.userId), {})[int(row.movieId)] = float(
                row.rating
            )

        self._knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._knn.fit(self.item_user_matrix)

    def _movie_id_for_title(self, title: str) -> int | None:
        return self.title_to_movie_id.get(str(title).strip().casefold())

    def _content_scores_for_movie(self, movie_id: int) -> np.ndarray:
        catalog_index = self.movie_id_to_catalog_index[movie_id]
        if self._genre_vectors.shape[1] == 0:
            return np.zeros(len(self.movies), dtype=float)
        return cosine_similarity(
            self._genre_vectors[catalog_index], self._genre_vectors
        ).ravel()

    def _knn_neighbors(self, movie_id: int, limit: int) -> list[tuple[int, float]]:
        if limit <= 0 or movie_id not in self.movie_id_to_knn_index:
            return []
        item_index = self.movie_id_to_knn_index[movie_id]
        neighbor_count = min(limit + 1, len(self.item_movie_ids))
        distances, indices = self._knn.kneighbors(
            self.item_user_matrix[item_index], n_neighbors=neighbor_count
        )

        results: list[tuple[int, float]] = []
        for distance, neighbor_index in zip(distances[0], indices[0]):
            neighbor_id = int(self.item_movie_ids[int(neighbor_index)])
            if neighbor_id == movie_id:
                continue
            similarity = max(0.0, 1.0 - float(distance))
            # Keep zero-similarity neighbors to preserve the original KNN
            # behavior, which returned the requested nearest items even when
            # their cosine distance was 1.
            results.append((neighbor_id, similarity))
            if len(results) >= limit:
                break
        return results

    def _collaborative_scores_for_movie(
        self, movie_id: int, top_n: int
    ) -> dict[int, float]:
        return dict(self._knn_neighbors(movie_id, top_n))

    def _hybrid_scores_for_movie(self, movie_id: int, top_n: int) -> np.ndarray:
        content_scores = self._content_scores_for_movie(movie_id)
        collaborative_scores = np.zeros(len(self.movies), dtype=float)
        for candidate_id, score in self._collaborative_scores_for_movie(
            movie_id, top_n
        ).items():
            catalog_index = self.movie_id_to_catalog_index.get(candidate_id)
            if catalog_index is not None:
                collaborative_scores[catalog_index] = score
        return 0.5 * content_scores + 0.5 * collaborative_scores

    def _titles_for_movie_ids(self, movie_ids: list[int], top_n: int) -> list[str]:
        titles: list[str] = []
        seen_titles: set[str] = set()
        for movie_id in movie_ids:
            title = self.movie_id_to_title.get(int(movie_id))
            if title and title.casefold() not in seen_titles:
                titles.append(title)
                seen_titles.add(title.casefold())
            if len(titles) >= top_n:
                break
        return titles

    def content_recommend(self, movie_title: str, top_n: int = 5) -> list[str]:
        movie_id = self._movie_id_for_title(movie_title)
        if movie_id is None:
            return ["Movie not found"]
        if top_n <= 0:
            return []

        scores = self._content_scores_for_movie(movie_id)
        seed_index = self.movie_id_to_catalog_index[movie_id]
        ranked_indices = np.argsort(-scores, kind="stable")
        movie_ids = [
            int(self.movies.iloc[index]["movieId"])
            for index in ranked_indices
            if index != seed_index
        ]
        return self._titles_for_movie_ids(movie_ids, top_n)

    def collab_recommend(self, movie_title: str, top_n: int = 5) -> list[str]:
        movie_id = self._movie_id_for_title(movie_title)
        if movie_id is None:
            return ["Movie not found"]
        if movie_id not in self.movie_id_to_knn_index:
            return ["Not enough data"]
        if top_n <= 0:
            return []

        neighbors = self._knn_neighbors(movie_id, top_n)
        return self._titles_for_movie_ids(
            [neighbor_id for neighbor_id, _ in neighbors], top_n
        )

    def hybrid_recommend(self, movie_title: str, top_n: int = 5) -> list[str]:
        movie_id = self._movie_id_for_title(movie_title)
        if movie_id is None:
            return ["Movie not found"]
        if top_n <= 0:
            return []

        scores = self._hybrid_scores_for_movie(movie_id, top_n)
        seed_index = self.movie_id_to_catalog_index[movie_id]
        scores[seed_index] = -np.inf
        ranked_indices = np.argsort(-scores, kind="stable")
        movie_ids = [int(self.movies.iloc[index]["movieId"]) for index in ranked_indices]
        return self._titles_for_movie_ids(movie_ids, top_n)

    def cold_start_recommend(self, genre: str, top_n: int = 5) -> list[str]:
        genre_pattern = str(genre).strip()
        if not genre_pattern or top_n <= 0:
            return []

        popularity = self.ratings.groupby("movieId").size().to_dict()
        # Preserve the original pandas substring/regex matching (e.g.
        # "Action" matches the "Action|Adventure" genre string).
        genre_matches = self.movies["genres"].str.contains(
            genre_pattern, case=False, na=False
        )
        matches = [
            (int(row.movieId), index)
            for index, row in enumerate(self.movies.itertuples(index=False))
            if genre_matches.iloc[index]
        ]
        matches.sort(key=lambda pair: (-popularity.get(pair[0], 0), pair[1]))
        return self._titles_for_movie_ids([movie_id for movie_id, _ in matches], top_n)

    def recommend_movie_ids_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        neighbors_per_item: int = 50,
    ) -> list[int]:
        """Rank unseen catalog items from one user's training interactions.

        Every user's item neighborhoods come from the same KNN model fit once
        on the interactions supplied to this recommender instance.
        """
        if top_n <= 0 or neighbors_per_item <= 0:
            return []
        known_ratings = self.user_ratings.get(int(user_id))
        if not known_ratings:
            return []

        seed_ids = [
            movie_id
            for movie_id in sorted(known_ratings)
            if movie_id in self.movie_id_to_knn_index
        ]
        if not seed_ids:
            return []

        seed_indices = [self.movie_id_to_knn_index[movie_id] for movie_id in seed_ids]
        query_matrix = self.item_user_matrix[seed_indices]
        neighbor_count = min(neighbors_per_item + 1, len(self.item_movie_ids))
        distances, indices = self._knn.kneighbors(
            query_matrix, n_neighbors=neighbor_count
        )

        seen_ids = set(known_ratings)
        weighted_scores: dict[int, float] = {}
        similarity_weights: dict[int, float] = {}
        for seed_id, distance_row, index_row in zip(seed_ids, distances, indices):
            preference = max(0.0, known_ratings[seed_id])
            for distance, neighbor_index in zip(distance_row, index_row):
                candidate_id = int(self.item_movie_ids[int(neighbor_index)])
                if candidate_id in seen_ids:
                    continue
                if candidate_id not in self.movie_id_to_catalog_index:
                    continue
                similarity = max(0.0, 1.0 - float(distance))
                if similarity <= 0:
                    continue
                weighted_scores[candidate_id] = (
                    weighted_scores.get(candidate_id, 0.0) + preference * similarity
                )
                similarity_weights[candidate_id] = (
                    similarity_weights.get(candidate_id, 0.0) + similarity
                )

        ranked = sorted(
            weighted_scores,
            key=lambda movie_id: (
                -(weighted_scores[movie_id] / similarity_weights[movie_id]),
                movie_id,
            ),
        )
        return ranked[:top_n]


def load_default_recommender() -> HybridMovieRecommender:
    """Load configured ratings, or the bundled small fixture for the Flask demo."""
    raw_limit = os.environ.get(RATINGS_MAX_ROWS_ENV)
    ratings_nrows = (
        DEFAULT_RATINGS_NROWS if raw_limit is None else parse_ratings_limit(raw_limit)
    )
    configured_ratings_path = os.environ.get(RATINGS_PATH_ENV)
    ratings_path = resolve_data_path(
        configured_ratings_path,
        env_var=RATINGS_PATH_ENV,
        default_name="ratings.csv",
    )
    if configured_ratings_path is None or not ratings_path.is_file():
        ratings_path = DEMO_RATINGS_PATH
    movies, ratings = load_data(
        ratings_path=ratings_path, ratings_nrows=ratings_nrows
    )
    return HybridMovieRecommender(movies, ratings)
