from utils.dataLoader import IMDbLoader, MovieLensLoader, MetadataPreprocessor, RatingsPreprocessor

class DataTester:
    def __init__(self):
        self.metadata = None
        self.featureMatrix = None
        self.ratings = None
        self.binaryRatings = None

    def run(self):
        # Load and preprocess movie metadata
        imdbLoader = IMDbLoader("ml-100k/links.csv")
        self.metadata = imdbLoader.loadMetadata()
        self.cleanedMetadata = imdbLoader.preprocessMetadata()
        print(f" Loaded {len(self.cleanedMetadata)} movies metadata")

        # Feature engineering
        metadataProcessor = MetadataPreprocessor(self.cleanedMetadata)
        categoricalFeatures = metadataProcessor.encodeCategoricalFeatures()
        voteFeatures = metadataProcessor.normalizeVoteAverage()

        # Combine features
        self.featureMatrix = categoricalFeatures.join(voteFeatures)
        self.featureMatrix.index = self.cleanedMetadata["movieId"]
        print(f" Feature matrix shape: {self.featureMatrix.shape}")

        # Load ratings
        movieLensLoader = MovieLensLoader("ml-100k/ratings.csv")
        self.ratings = movieLensLoader.loadRatings()

        # Preprocess ratings
        ratingsProcessor = RatingsPreprocessor(self.ratings)
        self.binaryRatings = ratingsProcessor.binarizeRatings(threshold=3.5)
        print(f" Loaded {self.ratings['userId'].nunique()} users and {self.ratings['movieId'].nunique()} movies in ratings")
        print(f" Ratings binarized: {self.binaryRatings['binaryRating'].value_counts().to_dict()}")

        return {
            "metadata": self.cleanedMetadata,
            "featureMatrix": self.featureMatrix,
            "binaryRatings": self.binaryRatings
        }
