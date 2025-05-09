import pandas as pd
import numpy as np
from utils.dataLoader import IMDbLoader, MovieLensLoader, MetadataPreprocessor, RatingsPreprocessor

class DataTester:
    def __init__(self):
        self.metadata = None
        self.featureMatrix = None
        self.ratings = None
        self.binaryRatings = None

    def run(self):
        print("\n Running Data Tester: \n")

        try:
            self._load_metadata()
            self._build_features()
            self._load_ratings()
            self._print_summary()

            return {
                "metadata": self.metadata,
                "featureMatrix": self.featureMatrix,
                "ratings": self.ratings,
                "binaryRatings": self.binaryRatings
            }

        except Exception as e:
            print(f"\n Error in DataTester: {str(e)}")
            return self._get_fallback_output()

    def _load_metadata(self):
        imdbLoader = IMDbLoader("ml-100k/links.csv", apiKey="766c1b0d")
        self.metadata = imdbLoader.loadMetadata()
        self.metadata = imdbLoader.preprocessMetadata()
        print(f" Loaded {len(self.metadata)} movies metadata")

    def _build_features(self):
        processor = MetadataPreprocessor(self.metadata)
        categorical = processor.encodeCategoricalFeatures()
        votes = processor.normalizeVoteAverage()
        self.featureMatrix = pd.concat([categorical, votes], axis=1)
        self.featureMatrix.index = self.metadata["movieId"]
        print(f" Feature matrix shape: {self.featureMatrix.shape}")

    def _load_ratings(self):
        loader = MovieLensLoader("ml-100k/ratings.csv")
        self.ratings = loader.loadRatings()
        processor = RatingsPreprocessor(self.ratings)
        self.binaryRatings = processor.binarizeRatings(threshold=3.5)
        print(f" Loaded ratings: {self.ratings.shape[0]} rows")

    def _print_summary(self):
        print("\n Summary:")
        print(f"  - Metadata: {len(self.metadata)} movies")
        print(f"  - Users: {self.ratings['userId'].nunique()}")
        print(f"  - Feature matrix: {self.featureMatrix.shape}")
        if 'binaryRating' in self.binaryRatings.columns:
            dist = self.binaryRatings['binaryRating'].value_counts(normalize=True).round(3).to_dict()
            print(f"  - Binary rating distribution: {dist}")
        print("\n Metadata sample:")
        print(self.metadata[['movieId', 'title']].head())

    def _get_fallback_output(self):
        return {
            "metadata": pd.DataFrame(columns=["movieId", "title"]),
            "featureMatrix": pd.DataFrame(),
            "ratings": pd.DataFrame(),
            "binaryRatings": pd.DataFrame()
        }

if __name__ == "__main__":
    tester = DataTester()
    results = tester.run()
