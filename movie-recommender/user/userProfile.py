import pandas as pd
from typing import List

class UserProfile:
    def __init__(self, userId: int):
        self.userId = userId
        self.favoriteMovies = []  # List of movie IDs
        self.feedbackHistory = {}  # {movieId: like/dislike}
        self.contentVector = None  # Aggregated vector from metadata
        self.collabVector = None   # Learned from matrix factorization

    # Add initial favorite movies during onboarding
    def addFavorites(self, movieIds: List[int]) -> None:
        pass

    # Record user feedback for a given movie
    def addFeedback(self, movieId: int, feedback: int) -> None:
        pass

    # Build content vector based on current favorites
    def buildContentVector(self, featureMatrix: pd.DataFrame) -> pd.Series:
        pass

    # Update collaborative vector using learned embeddings
    def updateCollaborativeVector(self, userFactors: pd.DataFrame) -> None:
        pass

    # Retrieve most recent state of the user profile
    def getProfileSummary(self) -> dict:
        pass
