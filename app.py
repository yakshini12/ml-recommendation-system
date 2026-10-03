"""Flask application for the hybrid movie recommender."""

import os

from flask import Flask, render_template, request

from model import HybridMovieRecommender, load_default_recommender


def create_app(recommender: HybridMovieRecommender | None = None) -> Flask:
    app = Flask(__name__)
    cached_recommender = recommender

    def get_recommender() -> HybridMovieRecommender:
        nonlocal cached_recommender
        if cached_recommender is None:
            cached_recommender = load_default_recommender()
        return cached_recommender

    @app.route("/", methods=["GET", "POST"])
    def home():
        if request.method == "GET":
            return render_template("index.html")

        movie = (request.form.get("movie") or "").strip()
        genre = (request.form.get("genre") or "").strip()
        if not movie and not genre:
            return render_template(
                "index.html", error="Enter a movie title or choose a genre."
            ), 400

        try:
            model = get_recommender()
        except (FileNotFoundError, ValueError) as error:
            return render_template("index.html", error=str(error)), 503

        if movie:
            return render_template(
                "index.html",
                content=model.content_recommend(movie),
                collab=model.collab_recommend(movie),
                hybrid=model.hybrid_recommend(movie),
                movie_name=movie,
                content_exp="Based on TF-IDF similarity of movie genres.",
                collab_exp="Based on cosine-distance neighbors in the item-user ratings matrix.",
                hybrid_exp="Combines content and collaborative similarity scores with equal 50/50 weights.",
            )

        return render_template(
            "index.html",
            cold=model.cold_start_recommend(genre),
            cold_exp="Ranked by the number of user/movie interactions in this genre.",
            genre=genre,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
