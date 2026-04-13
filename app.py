from flask import Flask, request, render_template
from model import content_recommend, collab_recommend, hybrid_recommend, cold_start_recommend

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        movie = request.form.get('movie')
        genre = request.form.get('genre')
        language = request.form.get('language')

        if movie:
            return render_template(
                'index.html',
                content=content_recommend(movie),
                collab=collab_recommend(movie),
                hybrid=hybrid_recommend(movie),
                movie_name=movie,

                # EXPLANATIONS
                content_exp="Based on movie genres and metadata similarity.",
                collab_exp="Based on user rating patterns and behavior.",
                hybrid_exp="Combines both approaches for better accuracy."
            )

        else:
            return render_template(
                'index.html',
                cold=cold_start_recommend(genre, language),
                cold_exp="Recommended based on genre preference and popularity.",
                genre=genre
            )

    return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug=True)