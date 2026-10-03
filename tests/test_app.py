import re

import model
from app import create_app


class StubRecommender:
    def content_recommend(self, movie, top_n=5):
        return [f"content for {movie}"]

    def collab_recommend(self, movie, top_n=5):
        return [f"collaborative for {movie}"]

    def hybrid_recommend(self, movie, top_n=5):
        return [f"hybrid for {movie}"]

    def cold_start_recommend(self, genre, top_n=5):
        return [f"popular {genre}"]


def _section_recommendations(html, start, end):
    section = html.split(start, 1)[1].split(end, 1)[0]
    return re.findall(r'<div class="[^"]+">([^<]+)</div>', section)


def test_application_serves_page_and_recommendations_with_injected_model():
    client = create_app(StubRecommender()).test_client()

    assert client.get("/").status_code == 200
    response = client.post("/", data={"movie": "Example Movie"})

    assert response.status_code == 200
    assert b"content for Example Movie" in response.data
    assert b"collaborative for Example Movie" in response.data
    assert b"hybrid for Example Movie" in response.data


def test_application_uses_demo_without_external_ratings(monkeypatch):
    monkeypatch.delenv("RATINGS_CSV_PATH", raising=False)
    monkeypatch.delenv("RATINGS_MAX_ROWS", raising=False)
    client = create_app().test_client()

    assert client.get("/").status_code == 200
    response = client.post("/", data={"movie": "Toy Story (1995)"})

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    content = _section_recommendations(
        html, "<h3>📌 Content-Based</h3>", "<h3>📌 Collaborative</h3>"
    )
    collaborative = _section_recommendations(
        html, "<h3>📌 Collaborative</h3>", "<h3>📌 Hybrid</h3>"
    )
    hybrid = _section_recommendations(html, "<h3>📌 Hybrid</h3>", "</body>")

    assert "Toy Story 2 (1999)" in content
    assert collaborative
    assert hybrid


def test_demo_recommendations_are_deterministic(monkeypatch):
    monkeypatch.delenv("RATINGS_CSV_PATH", raising=False)
    monkeypatch.delenv("RATINGS_MAX_ROWS", raising=False)

    first = create_app().test_client().post(
        "/", data={"movie": "Toy Story (1995)"}
    )
    second = create_app().test_client().post(
        "/", data={"movie": "Toy Story (1995)"}
    )

    assert first.status_code == second.status_code == 200
    assert first.data == second.data


def test_genre_cold_start_works_with_demo_ratings(monkeypatch):
    monkeypatch.delenv("RATINGS_CSV_PATH", raising=False)
    monkeypatch.delenv("RATINGS_MAX_ROWS", raising=False)
    client = create_app().test_client()

    response = client.post("/", data={"genre": "Animation"})

    assert response.status_code == 200
    assert b"Popular Animation Movies" in response.data
    assert b"Toy Story (1995)" in response.data


def test_configured_external_ratings_path_is_used(monkeypatch, tmp_path):
    ratings_path = tmp_path / "external-ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating\n1,1,5\n1,3114,4\n2,1,4\n2,3114,5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RATINGS_CSV_PATH", str(ratings_path))
    monkeypatch.delenv("RATINGS_MAX_ROWS", raising=False)

    recommender = model.load_default_recommender()

    assert recommender.ratings[["userId", "movieId", "rating"]].to_dict("records") == [
        {"userId": 1, "movieId": 1, "rating": 5},
        {"userId": 1, "movieId": 3114, "rating": 4},
        {"userId": 2, "movieId": 1, "rating": 4},
        {"userId": 2, "movieId": 3114, "rating": 5},
    ]
