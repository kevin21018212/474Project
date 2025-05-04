from utils.dataLoader import IMDbLoader, MovieLensLoader, MetadataPreprocessor, RatingsPreprocessor
from models.contentFilter import ContentBasedFilter
from models.collabFilter import CollaborativeFilter
from models.hybrid import HybridRecommender
from utils.userProfile import UserProfile
from utils.omdbFetcher import OmdbFetcher
from utils.helpers import precisionAtK, recallAtK
import pandas as pd

# Load metadata and ratings
def loadData():
    imdbLoader = IMDbLoader("ml-100k/links.csv", apiKey="766c1b0d")
    metadataDF = imdbLoader.loadMetadata()
    metadataDF = imdbLoader.preprocessMetadata()

    movielensLoader = MovieLensLoader("ml-100k/ratings.csv")
    ratingsDF = movielensLoader.loadRatings()

    return metadataDF, ratingsDF

# Preprocess metadata and ratings into feature matrices
def preprocessData(metadataDF, ratingsDF):
    metadataProcessor = MetadataPreprocessor(metadataDF)

    genres = metadataProcessor.encodeCategoricalFeatures()
    plots = metadataProcessor.applyTfidfToPlots()
    votes = metadataProcessor.normalizeVoteAverage()
    contentFeatures = pd.concat([genres, plots, votes], axis=1)

    ratingsProcessor = RatingsPreprocessor(ratingsDF)
    binaryRatings = ratingsProcessor.binarizeRatings()

    return contentFeatures, binaryRatings

# Initialize user profile from favorite movie IDs
def initializeUser() -> UserProfile:
    user = UserProfile(userId=1)
    user.addFavorites([1, 32, 50,1196,1120])  
    return user

# Train models and build feature matrices
def trainModels(metadataDF, ratingsDF, contentFeatures):
    contentModel = ContentBasedFilter(metadataDF)
    contentModel.featureMatrix = contentFeatures
    contentModel.movieIdToIndex = {mid: idx for idx, mid in enumerate(metadataDF["movieId"])}

    collabModel = CollaborativeFilter(numFactors=30)
    collabModel.trainModel(ratingsDF)

    hybridModel = HybridRecommender(contentModel, collabModel, alpha=0.5)

    return contentModel, collabModel, hybridModel

# Run recommendation pipeline
def runRecommendationPipeline(user: UserProfile, contentModel, collabModel, hybridModel, topN=10):
    fetcher = OmdbFetcher(apiKey="766c1b0d")

    print(f"\n🎬 User {user.userId}'s Favorite Movies:")
    for movieId in user.favorites:
        try:
            title = fetcher.getMovieTitle(movieId)
        except:
            title = "Unknown"
        print(f"❤️ {title} (movieId={movieId})")

    userProfile = contentModel.buildUserProfile(user.favorites)

    # Score vectors
    blendedScores = hybridModel.blendScores(user.userId, userProfile)
    contentSims = pd.Series(
        contentModel.featureMatrix @ userProfile,
        index=contentModel.featureMatrix.index
    )
    collabScores = pd.Series({
        movieId: collabModel.predictRating(user.userId, movieId)
        for movieId in contentModel.featureMatrix.index
        if movieId in collabModel.movieIdMapping
    })

    # Normalize scores
    contentSims = (contentSims - contentSims.min()) / (contentSims.max() - contentSims.min() + 1e-8)
    collabScores = (collabScores - collabScores.min()) / (collabScores.max() - collabScores.min() + 1e-8)

    # Top N by hybrid score
    topMovieIds = blendedScores.sort_values(ascending=False).head(topN).index.tolist()

    print(f"\n🎯 Top {topN} Hybrid Recommendations for User {user.userId}:\n")
    print(f"{'Rank':<5} {'Title':<40} {'Hybrid':>8} {'Content':>8} {'Collab':>8}")
    print("-" * 70)
    for i, movieId in enumerate(topMovieIds, 1):
        title = fetcher.getMovieTitle(movieId)
        hybridScore = blendedScores.get(movieId, 0)
        contentScore = contentSims.get(movieId, 0)
        collabScore = collabScores.get(movieId, 0)
        print(f"{i:<5} {title:<40} {hybridScore:>8.3f} {contentScore:>8.3f} {collabScore:>8.3f}")

    return topMovieIds


# Evaluation placeholder
def evaluateResults(recommendations, groundTruth, k=100):
    prec = precisionAtK(recommendations, groundTruth, k)
    rec = recallAtK(recommendations, groundTruth, k)
    print(f"\n📊 Evaluation Results (at K={k}):")
    print(f"Precision@{k}: {prec:.3f}")
    print(f"Recall@{k}:    {rec:.3f}")


# Main execution
def main():
    metadataDF, ratingsDF = loadData()
    contentFeatures, binaryRatings = preprocessData(metadataDF, ratingsDF)

    user = initializeUser()
    contentModel, collabModel, hybridModel = trainModels(metadataDF, binaryRatings, contentFeatures)

    recommendations = runRecommendationPipeline(user, contentModel, collabModel, hybridModel, topN=10)

    groundTruth = binaryRatings[binaryRatings["userId"] == user.userId]["movieId"].tolist()
    evaluateResults(recommendations, groundTruth)

if __name__ == "__main__":
    main()
