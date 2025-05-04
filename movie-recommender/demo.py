import pandas as pd
import numpy as np
from utils.dataLoader import IMDbLoader, MovieLensLoader, MetadataPreprocessor, RatingsPreprocessor
from models.contentFilter import ContentBasedFilter
from models.collabFilter import CollaborativeFilter
from models.hybrid import HybridRecommender
from utils.userProfile import UserProfile
from utils.omdbFetcher import OmdbFetcher
from utils.helpers import precisionAtK, recallAtK

def load_and_train():
    imdbLoader = IMDbLoader("ml-100k/links.csv", apiKey="766c1b0d")
    metadataDF = imdbLoader.loadMetadata()
    metadataDF = imdbLoader.preprocessMetadata()

    movielensLoader = MovieLensLoader("ml-100k/ratings.csv")
    ratingsDF = movielensLoader.loadRatings()

    metadataProcessor = MetadataPreprocessor(metadataDF)
    contentFeatures = pd.concat([
        metadataProcessor.encodeCategoricalFeatures(),
        metadataProcessor.applyTfidfToPlots(),
        metadataProcessor.normalizeVoteAverage()
    ], axis=1)

    ratingsProcessor = RatingsPreprocessor(ratingsDF)
    binaryRatings = ratingsProcessor.binarizeRatings()

    contentModel = ContentBasedFilter(metadataDF)
    contentModel.featureMatrix = contentFeatures
    contentModel.movieIdToIndex = {mid: idx for idx, mid in enumerate(metadataDF["movieId"])}

    collabModel = CollaborativeFilter(numFactors=30)
    collabModel.trainModel(binaryRatings)

    hybridModel = HybridRecommender(contentModel, collabModel, alpha=0.5)
    fetcher = OmdbFetcher(apiKey="766c1b0d")

    return metadataDF, contentModel, collabModel, hybridModel, fetcher, ratingsDF

def run_demo():
    print("\n🎥 Welcome to the Movie Recommender Demo!")

    metadataDF, contentModel, collabModel, hybridModel, fetcher, ratingsDF = load_and_train()
    user = UserProfile(userId=999)
    user_feedback_df = pd.DataFrame({
        "userId": pd.Series(dtype="int"),
        "movieId": pd.Series(dtype="int"),
        "rating": pd.Series(dtype="float")
    })
    all_liked_ids = set()

    while True:
        print("\n🍿 Enter some movies you like (by name, partial names allowed). Type 'done' to exit.")
        liked_titles = input("Favorites: ").strip()
        if liked_titles.lower() == "done":
            break

        matched_ids = []
        for title in liked_titles.split(","):
            title = title.strip().lower()
            match = metadataDF[metadataDF["title"].str.lower().str.contains(title)]
            if match.empty:
                new_id = fetcher.addMovieByTitle(title)
                if new_id:
                    metadataDF = pd.concat([metadataDF, pd.DataFrame([{
                        "movieId": new_id,
                        "title": title,
                        "genres": "Unknown"
                    }])], ignore_index=True)
                    matched_ids.append(new_id)
            else:
                matched_ids.extend(match["movieId"].tolist())

        if not matched_ids:
            print("❌ No matches found. Try again.")
            continue

        user.addFavorites(matched_ids)
        all_liked_ids.update(matched_ids)
        print("\n❤️ Your Favorites:")
        print(" | ".join([fetcher.getMovieTitle(mid) for mid in sorted(all_liked_ids)]))

        while True:
            userProfile = contentModel.buildUserProfile(user.favorites)
            blendedScores = hybridModel.blendScores(user.userId, userProfile)
            filteredScores = blendedScores.drop(index=user.favorites, errors='ignore')
            topMovieIds = filteredScores.sort_values(ascending=False).head(10).index.tolist()

            print("\n🎯 Top 10 Recommendations:")
            print(f"{'Rank':<5} {'Title':<40} {'Hybrid':>8} {'Content':>8} {'Collab':>8}")
            print("-" * 70)
            for i, movieId in enumerate(topMovieIds, 1):
                title = fetcher.getMovieTitle(movieId)
                contentScore = contentModel.featureMatrix.loc[movieId] @ userProfile if movieId in contentModel.featureMatrix.index else 0
                collabScore = collabModel.predictRating(user.userId, movieId) if movieId in collabModel.movieIdMapping else 0.0
                contentNorm = (contentScore - blendedScores.min()) / (blendedScores.max() - blendedScores.min() + 1e-8)
                collabNorm = (collabScore - blendedScores.min()) / (blendedScores.max() - blendedScores.min() + 1e-8)
                hybridScore = blendedScores.get(movieId, 0)
                print(f"{i:<5} {title:<40} {hybridScore:>8.3f} {contentNorm:>8.3f} {collabNorm:>8.3f}")

            print("\n👍👎 Which movies did you like from this list? Enter titles or press Enter to skip.")
            feedback_input = input("Liked: ").strip()
            if not feedback_input:
                break

            liked_ids = []
            for title in feedback_input.split(","):
                match = metadataDF[metadataDF["title"].str.lower().str.contains(title.strip().lower())]
                liked_ids.extend(match["movieId"].tolist())

            disliked_ids = set(topMovieIds) - set(liked_ids)
            all_liked_ids.update(liked_ids)

            print("\n📝 Feedback Summary:")
            if liked_ids:
                print("✅ Liked: " + " | ".join([fetcher.getMovieTitle(mid) for mid in sorted(all_liked_ids)]))
            if disliked_ids:
                print("❌ Disliked: " + " | ".join([fetcher.getMovieTitle(mid) for mid in disliked_ids]))

            for movieId in liked_ids:
                user_feedback_df = pd.concat([
                    user_feedback_df,
                    pd.DataFrame([{"userId": user.userId, "movieId": movieId, "rating": 5.0}])
                ])
                user.addFavorites([movieId])

            for movieId in disliked_ids:
                user_feedback_df = pd.concat([
                    user_feedback_df,
                    pd.DataFrame([{"userId": user.userId, "movieId": movieId, "rating": 1.0}])
                ])

            # Retrain collaborative model
            augmented_ratings = pd.concat([ratingsDF, user_feedback_df], ignore_index=True)
            collabModel.trainModel(augmented_ratings)

    print("\n📢 Thanks for trying the Movie Recommender Demo! Come back soon 🎬")

if __name__ == "__main__":
    run_demo()
