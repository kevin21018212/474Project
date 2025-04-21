import pandas as pd

class IMDbLoader:
    def __init__(self, metadataPath: str):
        self.metadataPath = metadataPath
        self.metadataDF = None

    # Load and return movie metadata from IMDb (genres, directors, actors)
    def loadMetadata(self) -> pd.DataFrame:
        pass

    # Clean, normalize, and extract useful fields from metadata
    def preprocessMetadata(self) -> pd.DataFrame:
        pass


class MovieLensLoader:
    def __init__(self, ratingsPath: str):
        self.ratingsPath = ratingsPath
        self.ratingsDF = None

    # Load user–movie ratings matrix from MovieLens 
    def loadRatings(self) -> pd.DataFrame:
        pass

    # Normalize ratings, handle missing values if needed
    def preprocessRatings(self) -> pd.DataFrame:
        pass
