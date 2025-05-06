# main.py
from utils.dataLoader import IMDbLoader, MovieLensLoader, MetadataPreprocessor, RatingsPreprocessor
from models.contentFilter import ContentBasedFilter
from models.collabFilter import CollaborativeFilter
from models.hybrid import HybridRecommender
from utils.userProfile import UserProfile
from utils.omdbFetcher import OmdbFetcher
from utils.helpers import precisionAtK, recallAtK
import pandas as pd

# Load metadata and ratings from files
def load_data():
    imdb = IMDbLoader("ml-100k/links.csv", apiKey="766c1b0d")
    metadata = imdb.loadMetadata()
    metadata = imdb.preprocessMetadata()

    movielens = MovieLensLoader("ml-100k/ratings.csv")
    ratings = movielens.loadRatings()

    return metadata, ratings

# Preprocess metadata into feature vectors, and binarize ratings
def preprocess(metadata, ratings):
    metaProc = MetadataPreprocessor(metadata)
    features = pd.concat([
        metaProc.encodeCategoricalFeatures(),    # One-hot encode genres, actors, directors
        metaProc.applyTfidfToPlots(),            # TF-IDF on plot summaries
        metaProc.normalizeVoteAverage()          # Normalize average rating
    ], axis=1)

    binRatings = RatingsPreprocessor(ratings).binarizeRatings()  # Convert ratings to binary like/dislike
    return features, binRatings

# Train all three models: content-based, collaborative, and hybrid
def train_models(metadata, ratings, features):
    content = ContentBasedFilter(metadata)
    content.featureMatrix = features
    content.movieIdToIndex = {mid: idx for idx, mid in enumerate(metadata["movieId"])}

    collab = CollaborativeFilter(numFactors=100)
    collab.trainModel(ratings)

    hybrid = HybridRecommender(content, collab, alpha=0.5)
    return content, collab, hybrid

# Recommend top movies based on hybrid score
def run_recommendation(user, content, collab, hybrid, topN=10):
    fetcher = OmdbFetcher(apiKey="766c1b0d")
    profile = content.buildUserProfile(user.favorites)
    hybrid_scores = hybrid.blendScores(user.userId, profile)

    # Drop any favorites already seen by user
    top_ids = hybrid_scores.drop(index=user.favorites, errors="ignore").sort_values(ascending=False).head(topN).index.tolist()

    print(f"\nTop {topN} recommendations:")
    for i, mid in enumerate(top_ids, 1):
        print(f"{i}. {fetcher.getMovieTitle(mid)}")

    return top_ids

# Evaluate precision and recall of recommendations
def evaluate(recs, truth, k=5):
    print(f"\nPrecision@{k}: {precisionAtK(recs, truth, k):.3f}")
    print(f"Recall@{k}:    {recallAtK(recs, truth, k):.3f}")

# Entry point for running the main model training and evaluation pipeline
def main():
    metadata, ratings = load_data()
    features, binRatings = preprocess(metadata, ratings)
    content, collab, hybrid = train_models(metadata, binRatings, features)

    user = UserProfile(userId=1)
    user.addFavorites([1, 32, 50, 1196, 1120])

    recs = run_recommendation(user, content, collab, hybrid)
    truth = binRatings[binRatings.userId == user.userId].movieId.tolist()
    evaluate(recs, truth)


if __name__ == "__main__":
    main()