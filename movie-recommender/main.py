from utils.dataLoader import IMDbLoader, MovieLensLoader
from utils.dataLoader import MetadataPreprocessor, RatingsPreprocessor
from models.contentFilter import ContentBasedRecommender
from models.collabFilter import CollaborativeRecommender
from models.hybrid import HybridRecommender
from utils.userProfile import UserProfile
from utils.metrics import computeRmse, computePrecisionAtK
from utils.dataLoader import IMDbLoader, MovieLensLoader, MetadataPreprocessor, RatingsPreprocessor

# Load metadata and ratings
def loadData():
    # Load OMDb metadata (cache or fetch)
    imdbLoader = IMDbLoader("ml-100k/links.csv")
    metadataDf = imdbLoader.loadMetadata()
    metadataDf = imdbLoader.preprocessMetadata()

    # Load MovieLens ratings
    movielensLoader = MovieLensLoader("ml-100k/ratings.csv")
    ratingsDf = movielensLoader.loadRatings()
    
    return metadataDf, ratingsDf

# Preprocess metadata and ratings
def preprocessData(metadataDf, ratingsDf):
    # Preprocess metadata into features
    metadataProcessor = MetadataPreprocessor(metadataDf)

    categories = metadataProcessor.encodeCategoricalFeatures()
    tfidfFeatures = metadataProcessor.applyTfidfToPlots()
    voteFeatures = metadataProcessor.normalizeVoteAverage()

    # Combine all features into content feature matrix
    contentFeatures = categories.join(tfidfFeatures).join(voteFeatures)

    # Binarize user ratings
    ratingsProcessor = RatingsPreprocessor(ratingsDf)
    binaryRatings = ratingsProcessor.binarizeRatings(threshold=3.5)

    return contentFeatures, binaryRatings



#  user profile with favorite movies
def initializeUser(userId: int, favoriteMovieIds: list) -> UserProfile:
    user = UserProfile(userId)
    user.addFavorites(favoriteMovieIds)
    return user

# Train content and collaborative models
def trainModels(metadataDF, ratingsDF):
    pass

# Run hybrid recommendation pipeline
def runRecommendationPipeline(user: UserProfile, contentModel, collabModel, hybridModel):
    pass

# Evaluate recommendations and print metrics
def evaluateResults(recommendations, groundTruth):
    pass

# Main execution flow
def main():
    pass

if __name__ == "__main__":
    main()