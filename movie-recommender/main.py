from utils.dataLoader import IMDbLoader, MovieLensLoader
from utils.dataLoader import MetadataPreprocessor, RatingsPreprocessor
from models.contentFilter import ContentBasedRecommender
from models.collabFilter import CollaborativeRecommender
from models.hybrid import HybridRecommender
from utils.userProfile import UserProfile
from utils.metrics import computeRmse, computePrecisionAtK

# Load datasets (IMDb metadata, MovieLens ratings)
def loadData():
    # Load OMDb metadata (cache or by fetching from OMDb API)
    imdbLoader = IMDbLoader("ml-100k/links.csv")
    metadataDf = imdbLoader.loadMetadata()
    metadataDf = imdbLoader.preprocessMetadata()

    # Load ratings from ratings.csv and normalize scores
    movielensLoader = MovieLensLoader("ml-100k/ratings.csv")
    ratingsDf = movielensLoader.loadRatings()
    ratingsDf = movielensLoader.preprocessRatings()
    
    return metadataDf, ratingsDf

# Preprocess metadata and ratings
def preprocessData(metadataDf, ratingsDf):
    
    # Create and apply metadata feature transformations
    metadataProcessor = MetadataPreprocessor(metadataDf)

    Categories = metadataProcessor.encodeCategories()

    # Convert movie plot text,normalize, and combine into a matrix
    tfidfFeatures = metadataProcessor.applyTfidfToPlots()
    voteFeatures = metadataProcessor.normalizeNumericalFeatures()
    contentFeatures = Categories.join(tfidfFeatures).join(voteFeatures)

    # Binarize user ratings
    ratingsProcessor = RatingsPreprocessor(ratingsDf)
    binaryRatings = ratingsProcessor.binarizeRatings(threshold=0.6)
    return contentFeatures, binaryRatings


# Setup user profile with favorite movies
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