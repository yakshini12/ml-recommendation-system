from pathlib import Path

import pandas as pd

import evaluate


FIXTURES = Path(__file__).parent / "fixtures"


def fixture_data():
    return (
        pd.read_csv(FIXTURES / "movies.csv"),
        pd.read_csv(FIXTURES / "ratings.csv"),
    )


def test_per_user_split_has_no_interaction_overlap_and_keeps_training_rows():
    _, ratings = fixture_data()
    ratings = pd.concat([ratings, ratings.iloc[[0]]], ignore_index=True)

    train, holdout = evaluate.per_user_holdout_split(
        ratings, test_fraction=0.25, random_state=11
    )

    train_pairs = set(zip(train["userId"], train["movieId"]))
    holdout_pairs = set(zip(holdout["userId"], holdout["movieId"]))
    assert train_pairs.isdisjoint(holdout_pairs)
    assert train.groupby("userId").size().min() >= 1
    assert train_pairs | holdout_pairs == set(
        zip(ratings["userId"], ratings["movieId"])
    )


def test_holdout_evaluation_fits_one_shared_training_only_knn(monkeypatch):
    movies, ratings = fixture_data()
    train, holdout = evaluate.per_user_holdout_split(
        ratings, test_fraction=0.2, random_state=42
    )
    real_recommender = evaluate.HybridMovieRecommender
    model_training_sets = []

    def create_model(movie_frame, training_ratings):
        model_training_sets.append(training_ratings.copy())
        return real_recommender(movie_frame, training_ratings)

    monkeypatch.setattr(evaluate, "HybridMovieRecommender", create_model)
    result = evaluate.evaluate_per_user_holdout(
        movies, ratings, k=3, test_fraction=0.2, random_state=42
    )

    assert len(model_training_sets) == 1
    assert set(zip(model_training_sets[0]["userId"], model_training_sets[0]["movieId"])) == set(
        zip(train["userId"], train["movieId"])
    )
    assert set(zip(model_training_sets[0]["userId"], model_training_sets[0]["movieId"])).isdisjoint(
        set(zip(holdout["userId"], holdout["movieId"]))
    )
    assert result.evaluated_users > 0
    assert 0 <= result.precision_at_k <= 1
    assert 0 <= result.recall_at_k <= 1
