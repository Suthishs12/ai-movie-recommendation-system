import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# LOAD DATA
# -------------------------------
movies = pd.read_csv(r"C:\project alfido\Dataset\movies.csv")
ratings = pd.read_csv(r"C:\project alfido\Dataset\ratings.csv")

# -------------------------------
# PREPROCESS DATA
# -------------------------------
data = pd.merge(ratings, movies, on="movieId")
data = data.drop(columns=["timestamp"])

# Create user-movie matrix
user_movie_matrix = data.pivot_table(
    index="userId",
    columns="title",
    values="rating"
)

# Fill missing values with 0
user_movie_matrix_filled = user_movie_matrix.fillna(0)

# -------------------------------
# COSINE SIMILARITY
# -------------------------------
movie_similarity = cosine_similarity(user_movie_matrix_filled.T)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix_filled.columns,
    columns=user_movie_matrix_filled.columns
)

# -------------------------------
# RECOMMENDATION FUNCTION
# -------------------------------
def recommend_movies(movie_name, top_n=5):
    if movie_name not in movie_similarity_df.columns:
        return []

    similarity_scores = movie_similarity_df[movie_name]
    similarity_scores = similarity_scores.sort_values(ascending=False)
    similarity_scores = similarity_scores.drop(movie_name)

    return similarity_scores.head(top_n).index.tolist()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="AI Movie Recommendation System", layout="centered")

st.title("🎬 AI Movie Recommendation System")
st.write("Select a movie and get similar movie recommendations")

movie_list = sorted(movie_similarity_df.columns.tolist())

selected_movie = st.selectbox("🎥 Select a Movie", movie_list)

num_recommendations = st.slider(
    "🔢 Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("🎯 Recommend"):
    recommendations = recommend_movies(selected_movie, num_recommendations)

    if recommendations:
        st.subheader("✅ Recommended Movies")
        for movie in recommendations:
            st.write("👉", movie)
    else:
        st.warning("No recommendations found.")
