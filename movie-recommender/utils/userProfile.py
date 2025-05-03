import pandas as pd
from typing import List
class UserProfile:
    def __init__(self, userId):  # Changed from user_id to userId
        self.userId = userId
        self.favorites = []
    
    def addFavorites(self, movie_ids):
        self.favorites.extend(movie_ids)
    
    def get_favorite_movies(self):
        return self.favorites

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
