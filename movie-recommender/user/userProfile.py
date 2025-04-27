import pandas as pd
from typing import List
class UserProfile:
    def __init__(self, userId: int):
        self.userId = userId
        self.favoriteMovies = []       # List of movie IDs
        self.feedbackHistory = {}      # {movieId: like/dislike (1/0)}
        self.contentVector = None      # Aggregated feature vector
        self.collabVector = None       # Latent factors from matrix factorization

    # Add initial favorite movies 
    def addFavorites(self, movieIds: List[int]) -> None:
        self.favoriteMovies.extend(movieIds)

    # Record feedback for movie
    def addFeedback(self, movieId: int, feedback: int) -> None:
        self.feedbackHistory[movieId] = feedback

    # Build vector based on current favorites
    def buildContentVector(self, featureMatrix: pd.DataFrame) -> pd.Series:
        if not self.favoriteMovies:
            raise ValueError("No favorite movies to build content vector.")
        
        # Filter feature matrix to only favorite movies
        fav_features = featureMatrix.loc[self.favoriteMovies]
        
        # Take mean of favorite movie features
        self.contentVector = fav_features.mean(axis=0)
        return self.contentVector

    # Update using learned user factors
    def updateCollaborativeVector(self, userFactors: pd.DataFrame) -> None:
        if self.userId not in userFactors.index:
            raise ValueError(f"User {self.userId} not found in collaborative model factors.")
        
        self.collabVector = userFactors.loc[self.userId]

    # Get user profile
    def getProfileSummary(self) -> dict:
        return {
            "userId": self.userId,
            "favorites": self.favoriteMovies,
            "feedbackHistory": self.feedbackHistory,
            "contentVectorShape": None if self.contentVector is None else self.contentVector.shape,
            "collabVectorShape": None if self.collabVector is None else self.collabVector.shape
        }
