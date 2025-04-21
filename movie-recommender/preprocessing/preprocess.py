import pandas as pd

class MetadataPreprocessor:
    def __init__(self, metadataDF: pd.DataFrame):
        self.metadataDF = metadataDF

    # One-hot encode genres, directors, actors
    def encodeCategoricalFeatures(self) -> pd.DataFrame:
        pass

    # Apply TF-IDF vectorization on plot summaries
    def applyTfidfToPlots(self) -> pd.DataFrame:
        pass

    # Normalize numerical fields (e.g., IMDb rating)
    def normalizeNumericalFeatures(self) -> pd.DataFrame:
        pass


class RatingsPreprocessor:
    def __init__(self, ratingsDF: pd.DataFrame):
        self.ratingsDF = ratingsDF

    # Normalize ratings to a consistent scale (optional)
    def normalizeRatings(self) -> pd.DataFrame:
        pass

    # Convert ratings to binary labels (like/dislike)
    def binarizeRatings(self, threshold: float = 3.0) -> pd.DataFrame:
        pass

    # Handle missing values in ratings dataset
    def fillMissingValues(self) -> pd.DataFrame:
        pass
