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


def test_application_serves_page_and_recommendations_with_injected_model():
    client = create_app(StubRecommender()).test_client()

    assert client.get("/").status_code == 200
    response = client.post("/", data={"movie": "Example Movie"})

    assert response.status_code == 200
    assert b"content for Example Movie" in response.data
    assert b"collaborative for Example Movie" in response.data
    assert b"hybrid for Example Movie" in response.data


def test_application_reports_missing_ratings_file(monkeypatch, tmp_path):
    monkeypatch.setenv("RATINGS_CSV_PATH", str(tmp_path / "missing-ratings.csv"))
    client = create_app().test_client()

    assert client.get("/").status_code == 200
    response = client.post("/", data={"movie": "Toy Story (1995)"})

    assert response.status_code == 503
    assert b"Ratings data not found" in response.data
