
import pandas as pd
from typing import List

class ContentBasedFilter:
    def __init__(self, metadataDF: pd.DataFrame):
        self.metadataDF = metadataDF
        self.featureMatrix = None

    # Build feature matrix from metadata (genres, director, actors)
    def buildFeatureMatrix(self) -> None:
        pass

    # Generate content vector for a user based on their favorite movies
    def buildUserProfile(self, favoriteMovieIds: List[int]) -> pd.Series:
        pass

    # Recommend movies similar to user profile based on cosine similarity
    def recommendMovies(self, userProfile: pd.Series, topN: int = 10) -> List[int]:
        pass

    # Optionally update profile based on new likes/dislikes
    def updateUserProfile(self, userProfile: pd.Series, movieId: int, feedback: int) -> pd.Series:
        pass
