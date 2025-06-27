import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Create a Sample Dataset (User-Item Ratings) ---
# Ratings are on a scale of 1 to 5, NaN indicates an unrated movie.
data = {
    'User': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Charlie', 'Charlie', 'David', 'David', 'Eve', 'Eve', 'Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie A', 'Movie D', 'Movie B', 'Movie E', 'Movie C', 'Movie D', 'Movie A', 'Movie E', 'Movie D', 'Movie C', 'Movie F', 'Movie E', 'Movie F'],
    'Rating': [5, 4, np.nan, 4, 5, 5, 4, 3, np.nan, 5, 5, 2, 4, 5, 4, np.nan]
}
df = pd.DataFrame(data)

print("Original Ratings Dataset:")
print(df)
print("\n")

# --- 2. Create User-Item Matrix ---
# Pivot the DataFrame to get users as rows, movies as columns, and ratings as values.
# Fill NaN for movies not rated by a user.
user_movie_matrix = df.pivot_table(index='User', columns='Movie', values='Rating')

print("User-Movie Matrix (NaN for unrated movies):")
print(user_movie_matrix)
print("\n")

# --- 3. Calculate User Similarity (Cosine Similarity) ---
# Fill NaN with 0 for similarity calculation. This is a common approach,
# but it implies that unrated items mean 'no preference', not 'dislike'.
# For more advanced scenarios, mean centering or other imputation might be used.
user_movie_matrix_filled = user_movie_matrix.fillna(0)

# Calculate cosine similarity between users.
# We subtract from 1 because pairwise_distances returns distance, not similarity.
# cosine_similarity = 1 - cosine_distance
user_similarity = 1 - pairwise_distances(user_movie_matrix_filled, metric='cosine')
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

print("User Similarity Matrix (Cosine Similarity):")
print(user_similarity_df)
print("\n")

# --- 4. Generate Recommendations for a Target User ---
target_user = 'Alice'
num_recommendations = 3

print(f"Generating recommendations for: {target_user}")

# Get similarity scores for the target user with all other users
target_user_similarities = user_similarity_df[target_user].sort_values(ascending=False)

# Remove the target user themselves from the similarity list
target_user_similarities = target_user_similarities.drop(target_user)

print(f"\nUsers most similar to {target_user}:")
print(target_user_similarities)

# Get the top similar users (e.g., top N, or users above a certain threshold)
# For simplicity, we consider users with similarity > 0.1
similar_users = target_user_similarities[target_user_similarities > 0.1].index

if similar_users.empty:
    print(f"\nNo sufficiently similar users found for {target_user}.")
else:
    print(f"\nFound similar users: {list(similar_users)}")

    # Get movies rated by the target user
    movies_rated_by_target = user_movie_matrix.loc[target_user].dropna().index.tolist()
    print(f"Movies rated by {target_user}: {movies_rated_by_target}")

    recommendations = {} # Dictionary to store predicted ratings for unrated movies

    # Iterate through each movie in the dataset
    for movie in user_movie_matrix.columns:
        if movie not in movies_rated_by_target: # Only consider movies not rated by the target user
            weighted_sum = 0
            similarity_sum = 0

            # Iterate through similar users
            for user in similar_users:
                # Check if the similar user has rated this movie
                if not pd.isna(user_movie_matrix.loc[user, movie]):
                    rating_by_similar_user = user_movie_matrix.loc[user, movie]
                    similarity_score = user_similarity_df.loc[target_user, user]

                    weighted_sum += rating_by_similar_user * similarity_score
                    similarity_sum += similarity_score

            if similarity_sum > 0: # Avoid division by zero
                predicted_rating = weighted_sum / similarity_sum
                recommendations[movie] = predicted_rating

    # Sort recommendations by predicted rating in descending order
    sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)

    print(f"\nTop {num_recommendations} Recommendations for {target_user}:")
    if sorted_recommendations:
        for movie, rating in sorted_recommendations[:num_recommendations]:
            print(f"- {movie} (Predicted Rating: {rating:.2f})")
    else:
        print("No new recommendations could be generated.")

# --- 5. Visualize User Similarity Matrix (Heatmap) ---
plt.figure(figsize=(8, 6))
sns.heatmap(user_similarity_df, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('User Similarity Matrix (Cosine Similarity)', fontsize=16)
plt.xlabel('User', fontsize=12)
plt.ylabel('User', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save the plot
heatmap_path = 'user_similarity_heatmap.png'
plt.savefig(heatmap_path)
plt.close() # Close the plot to free memory
print(f"\nUser similarity heatmap saved to '{heatmap_path}'")
