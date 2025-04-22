from utils.dataLoader import IMDbLoader, MovieLensLoader, MetadataPreprocessor, RatingsPreprocessor

class DataTester:
    def __init__(self):
        self.metadata = None
        self.cleaned_metadata = None
        self.cat_features = None
        self.tfidf_features = None
        self.vote_features = None
        self.ratings = None
        self.processed_ratings = None
        self.binarized_ratings = None

    def run(self):


        # === Load & preprocess metadata ===
        imdb_loader = IMDbLoader("ml-100k/links.csv")
        self.metadata = imdb_loader.loadMetadata()
        self.cleaned_metadata = imdb_loader.preprocessMetadata()

        print("📽️ Cleaned Metadata Sample:")
        print(self.cleaned_metadata[["movie_id", "title", "genres", "directors", "actors", "vote_average"]].head(10))
        print(f"🧮 Total movies loaded: {len(self.cleaned_metadata)}")

        # === Feature engineering ===
        meta_proc = MetadataPreprocessor(self.cleaned_metadata)
        self.cat_features = meta_proc.encodeCategoricalFeatures()
        self.tfidf_features = meta_proc.applyTfidfToPlots()
        self.vote_features = meta_proc.normalizeNumericalFeatures()

        print("\n🎨 Categorical Features Shape:", self.cat_features.shape)
        print("🧠 TF-IDF Vector Shape:", self.tfidf_features.shape)
        print("📊 Vote Feature Shape:", self.vote_features.shape)

        # === Load & preprocess ratings ===
        ml_loader = MovieLensLoader("ml-100k/ratings.csv")
        self.ratings = ml_loader.loadRatings()
        self.processed_ratings = ml_loader.preprocessRatings()

        print("\n⭐ Ratings Sample (Normalized):")
        print(self.processed_ratings.head(10))
        print(f"👥 Total users: {self.processed_ratings['userId'].nunique()}")
        print(f"🎞️ Total rated movies: {self.processed_ratings['movieId'].nunique()}")

        # === Binarized ratings ===
        ratings_proc = RatingsPreprocessor(self.processed_ratings)
        self.binarized_ratings = ratings_proc.binarizeRatings(threshold=0.6)

        print("\n✅ Binary Ratings Sample:")
        print(self.binarized_ratings.head(10))
        print("📈 Ratings value counts:")
        print(self.binarized_ratings["binary_rating"].value_counts())

        return {
            "metadata": self.cleaned_metadata,
            "cat_features": self.cat_features,
            "tfidf": self.tfidf_features,
            "votes": self.vote_features,
            "ratings": self.processed_ratings,
            "binary_ratings": self.binarized_ratings
        }
