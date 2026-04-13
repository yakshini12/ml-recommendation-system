import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# LOAD DATA
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv", nrows=200000)

# CLEAN
movies['title'] = movies['title'].str.strip()
movies['genres'] = movies['genres'].fillna('')
movies['language'] = 'English'  # dummy column

# CONTENT BASED
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
content_sim = cosine_similarity(tfidf_matrix)

# COLLABORATIVE
user_movie_matrix = ratings.pivot_table(
    index='userId', columns='movieId', values='rating'
).fillna(0)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix.T)

# CONTENT FUNCTION
def content_recommend(movie_title, top_n=5):
    if movie_title not in movies['title'].values:
        return ["Movie not found"]

    idx = movies[movies['title'] == movie_title].index[0]
    scores = list(enumerate(content_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return [movies.iloc[i[0]]['title'] for i in scores[1:top_n+1]]

# COLLAB FUNCTION
def collab_recommend(movie_title, top_n=5):
    if movie_title not in movies['title'].values:
        return ["Movie not found"]

    idx = movies[movies['title'] == movie_title].index[0]
    movie_id = movies.iloc[idx]['movieId']

    if movie_id not in user_movie_matrix.columns:
        return ["Not enough data"]

    movie_idx = list(user_movie_matrix.columns).index(movie_id)

    distances, indices = model_knn.kneighbors(
        user_movie_matrix.T.iloc[movie_idx].values.reshape(1, -1),
        n_neighbors=top_n+1
    )

    return [movies.iloc[i]['title'] for i in indices.flatten()[1:]]

# HYBRID FUNCTION
def hybrid_recommend(movie_title, top_n=5):
    if movie_title not in movies['title'].values:
        return ["Movie not found"]

    idx = movies[movies['title'] == movie_title].index[0]

    content_scores = content_sim[idx]

    collab_scores = np.zeros(len(movies))

    movie_id = movies.iloc[idx]['movieId']
    if movie_id in user_movie_matrix.columns:
        movie_idx = list(user_movie_matrix.columns).index(movie_id)

        distances, indices = model_knn.kneighbors(
            user_movie_matrix.T.iloc[movie_idx].values.reshape(1, -1),
            n_neighbors=top_n+1
        )

        for i, d in zip(indices.flatten(), distances.flatten()):
            collab_scores[i] = 1 - d

    hybrid_scores = 0.5 * content_scores + 0.5 * collab_scores

    top_indices = hybrid_scores.argsort()[::-1][1:top_n+1]

    return movies.iloc[top_indices]['title'].tolist()

# COLD START FUNCTION
def cold_start_recommend(genre, language='English', top_n=5):
    filtered = movies[movies['genres'].str.contains(genre, case=False, na=False)]
    popularity = ratings.groupby('movieId').size().sort_values(ascending=False)

    filtered['popularity'] = filtered['movieId'].map(popularity)
    filtered = filtered.sort_values(by='popularity', ascending=False)

    return filtered['title'].head(top_n).tolist()