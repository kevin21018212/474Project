# demo.py
from main import load_data, preprocess, train_models
from utils.userProfile import UserProfile
from utils.omdbFetcher import OmdbFetcher
import pandas as pd


def run_demo():
    print("\n🎥 Welcome to the Movie Recommender Demo!")

    # Load and preprocess data
    metadata, ratings = load_data()
    features, binRatings = preprocess(metadata, ratings)
    content, collab, hybrid = train_models(metadata, binRatings, features)
    fetcher = OmdbFetcher(apiKey="766c1b0d")

    user = UserProfile(userId=999)
    feedback = pd.DataFrame(columns=["userId", "movieId", "rating"])
    all_favs = set()

    while True:
        # Ask user for favorite movies
        query = input("\n🍿 Enter favorite movie titles (comma separated) or 'done': ").strip()
        if query.lower() == "done":
            break

        matched_ids = []
        for title in query.split(","):
            title = title.strip().lower()
            match = metadata[metadata["title"].str.lower().str.contains(title)]
            if not match.empty:
                matched_ids.extend(match["movieId"].tolist())

        if not matched_ids:
            print("❌ No matches found. Try again.")
            continue

        # Update user profile
        user.addFavorites(matched_ids)
        all_favs.update(matched_ids)

        print("\n❤️ Favorites so far:")
        print(" | ".join([fetcher.getMovieTitle(mid) for mid in sorted(all_favs)]))

        # Generate recommendations
        profile = content.buildUserProfile(user.favorites)
        scores = hybrid.blendScores(user.userId, profile).drop(index=user.favorites, errors="ignore")
        top_ids = scores.sort_values(ascending=False).head(10).index.tolist()

        print("\n🎯 Recommendations:")
        for i, mid in enumerate(top_ids, 1):
            print(f"{i}. {fetcher.getMovieTitle(mid)}")

        # Get feedback from user
        likes = input("\n👍 Enter titles you liked from above (or press Enter to skip): ").strip()
        if not likes:
            continue

        liked_ids = []
        for title in likes.split(","):
            title = title.strip().lower()
            match = metadata[metadata["title"].str.lower().str.contains(title)]
            liked_ids.extend(match["movieId"].tolist())

        # Add feedback to dataset
        for mid in liked_ids:
            feedback = pd.concat([feedback, pd.DataFrame([{"userId": user.userId, "movieId": mid, "rating": 5.0}])])
            user.addFavorites([mid])
            all_favs.add(mid)

        for mid in set(top_ids) - set(liked_ids):
            feedback = pd.concat([feedback, pd.DataFrame([{"userId": user.userId, "movieId": mid, "rating": 1.0}])])

        # Retrain collaborative model with updated feedback
        collab.trainModel(pd.concat([ratings, feedback], ignore_index=True))

    print("\n📢 Thanks for using the demo! 🎬")


if __name__ == "__main__":
    run_demo()