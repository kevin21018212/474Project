from utils.dataLoader import IMDbLoader, MovieLensLoader, MetadataPreprocessor, RatingsPreprocessor

class DataTester:
    def __init__(self):
        self.metadata = None
        self.featureMatrix = None
        self.ratings = None
        self.binaryRatings = None

    def run(self):
        # Load and preprocess movie metadata
        imdb_loader = IMDbLoader("ml-100k/links.csv")
        self.metadata = imdb_loader.loadMetadata()
        self.cleaned_metadata = imdb_loader.preprocessMetadata()

        # Feature engineering (only important parts)
        meta_proc = MetadataPreprocessor(self.cleaned_metadata)
        cat_features = meta_proc.encodeCategoricalFeatures()
        vote_features = meta_proc.normalizeNumericalFeatures()

        # Combine important features into one feature matrix
        self.featureMatrix = cat_features.join(vote_features)
        self.featureMatrix.index = self.cleaned_metadata["movie_id"]

        # Load and preprocess ratings
        ml_loader = MovieLensLoader("ml-100k/ratings.csv")
        self.ratings = ml_loader.loadRatings()
        processed_ratings = ml_loader.preprocessRatings()

        # Binarize ratings
        ratings_proc = RatingsPreprocessor(processed_ratings)
        self.binaryRatings = ratings_proc.binarizeRatings(threshold=0.6)

        return {
            "metadata": self.cleaned_metadata,
            "featureMatrix": self.featureMatrix,
            "binaryRatings": self.binaryRatings
        }
