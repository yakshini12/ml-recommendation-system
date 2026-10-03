"""Leakage-free per-user holdout evaluation for the item-based recommender."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from model import HybridMovieRecommender, load_data, normalize_ratings, parse_ratings_limit


@dataclass(frozen=True)
class EvaluationResult:
    precision_at_k: float
    recall_at_k: float
    k: int
    evaluated_users: int
    skipped_users: int
    train_interactions: int
    holdout_interactions: int
    test_fraction: float
    relevance_threshold: float
    random_state: int


def per_user_holdout_split(
    ratings: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out interactions per user while retaining at least one train row.

    Duplicate user/movie rows are averaged before splitting, so an interaction
    cannot occur in both the training and holdout sets.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be greater than 0 and less than 1.")

    cleaned = normalize_ratings(ratings)
    rng = np.random.default_rng(random_state)
    train_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []

    for _, user_rows in cleaned.groupby("userId", sort=True):
        row_count = len(user_rows)
        if row_count < 2:
            train_parts.append(user_rows)
            continue

        holdout_count = min(row_count - 1, max(1, math.ceil(row_count * test_fraction)))
        held_out_indices = rng.choice(user_rows.index.to_numpy(), size=holdout_count, replace=False)
        held_out_mask = user_rows.index.isin(held_out_indices)
        train_parts.append(user_rows.loc[~held_out_mask])
        holdout_parts.append(user_rows.loc[held_out_mask])

    train = pd.concat(train_parts, ignore_index=True)
    holdout = (
        pd.concat(holdout_parts, ignore_index=True)
        if holdout_parts
        else cleaned.iloc[0:0].copy()
    )
    train_pairs = set(zip(train["userId"], train["movieId"]))
    holdout_pairs = set(zip(holdout["userId"], holdout["movieId"]))
    if not train_pairs.isdisjoint(holdout_pairs):
        raise RuntimeError("A user/movie interaction leaked across the holdout split.")
    return train, holdout


def evaluate_per_user_holdout(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    k: int = 10,
    test_fraction: float = 0.2,
    relevance_threshold: float = 4.0,
    random_state: int = 42,
) -> EvaluationResult:
    """Fit one shared KNN on all training rows and score users' held-out items."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    train, holdout = per_user_holdout_split(
        ratings, test_fraction=test_fraction, random_state=random_state
    )
    if train.empty:
        raise ValueError("The training split is empty; evaluation needs training rows.")

    # This is the sole model/KNN fit in the evaluation. All users are scored
    # against neighborhoods learned from this shared training-only matrix.
    recommender = HybridMovieRecommender(movies, train)
    candidate_ids = set(recommender.movie_id_to_knn_index).intersection(
        recommender.movie_id_to_catalog_index
    )
    heldout_by_user = {
        int(user_id): rows
        for user_id, rows in holdout.groupby("userId", sort=True)
    }

    precision_values: list[float] = []
    recall_values: list[float] = []
    skipped_users = 0
    for user_id in sorted(recommender.user_ratings):
        user_holdout = heldout_by_user.get(user_id)
        if user_holdout is None:
            skipped_users += 1
            continue
        relevant_ids = {
            int(row.movieId)
            for row in user_holdout.itertuples(index=False)
            if float(row.rating) >= relevance_threshold and int(row.movieId) in candidate_ids
        }
        if not relevant_ids:
            skipped_users += 1
            continue

        recommendations = recommender.recommend_movie_ids_for_user(user_id, top_n=k)
        if set(recommendations).intersection(recommender.user_ratings[user_id]):
            raise RuntimeError("A training-seen movie was returned as a recommendation.")
        hits = len(relevant_ids.intersection(recommendations))
        precision_values.append(hits / k)
        recall_values.append(hits / len(relevant_ids))

    if not precision_values:
        raise ValueError(
            "No users have a relevant held-out movie that is present in both the "
            "training item set and the movie catalog. Check the data and relevance threshold."
        )

    return EvaluationResult(
        precision_at_k=float(np.mean(precision_values)),
        recall_at_k=float(np.mean(recall_values)),
        k=k,
        evaluated_users=len(precision_values),
        skipped_users=skipped_users,
        train_interactions=len(train),
        holdout_interactions=len(holdout),
        test_fraction=test_fraction,
        relevance_threshold=relevance_threshold,
        random_state=random_state,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate item-based recommendations with a per-user holdout."
    )
    parser.add_argument("--movies", help="Movie catalog CSV (defaults to movies.csv).")
    parser.add_argument("--ratings", help="Ratings CSV (defaults to ratings.csv).")
    parser.add_argument(
        "--ratings-rows",
        type=parse_ratings_limit,
        default=None,
        metavar="N|full",
        help="Ratings rows to load (default: all rows; use a positive integer to cap).",
    )
    parser.add_argument("--k", type=int, default=10, help="Recommendation list size.")
    parser.add_argument(
        "--test-fraction", type=float, default=0.2, help="Per-user holdout fraction."
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Minimum held-out rating counted as relevant.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        movies, ratings = load_data(
            args.movies, args.ratings, ratings_nrows=args.ratings_rows
        )
        result = evaluate_per_user_holdout(
            movies,
            ratings,
            k=args.k,
            test_fraction=args.test_fraction,
            relevance_threshold=args.relevance_threshold,
            random_state=args.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
