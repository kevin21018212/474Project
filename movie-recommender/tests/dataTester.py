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
        print(f"✅ Loaded {len(self.cleaned_metadata)} movies metadata")

        # Feature engineering
        meta_proc = MetadataPreprocessor(self.cleaned_metadata)
        cat_features = meta_proc.encodeCategoricalFeatures()
        vote_features = meta_proc.normalizeVoteAverage()

        # Combine features
        self.featureMatrix = cat_features.join(vote_features)
        self.featureMatrix.index = self.cleaned_metadata["movie_id"]
        print(f"✅ Feature matrix shape: {self.featureMatrix.shape}")

        # Load ratings
        ml_loader = MovieLensLoader("ml-100k/ratings.csv")
        self.ratings = ml_loader.loadRatings()

        # Preprocess ratings
        ratings_proc = RatingsPreprocessor(self.ratings)
        self.binaryRatings = ratings_proc.binarizeRatings(threshold=3.5)
        print(f"✅ Loaded {self.ratings['userId'].nunique()} users and {self.ratings['movieId'].nunique()} movies in ratings")
        print(f"✅ Ratings binarized: {self.binaryRatings['binary_rating'].value_counts().to_dict()}")

        return {
            "metadata": self.cleaned_metadata,
            "featureMatrix": self.featureMatrix,
            "binaryRatings": self.binaryRatings
        }
