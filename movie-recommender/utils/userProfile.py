import pandas as pd
from typing import List

class UserProfile:
    def __init__(self, userId):
        # Each user has a unique ID and a list of their favorite movies
        self.userId = userId
        self.favorites = []
        self.feedbackHistory = {}       # Feedback per movie (liked/disliked)
        self.contentVector = None       # Averaged vector from favorite movies
        self.collabVector = None        # Latent vector from collaborative filtering

    def addFavorites(self, movie_ids: List[int]):
        # Add one or more favorite movies
        self.favorites.extend(movie_ids)

    def get_favorite_movies(self) -> List[int]:
        # Return list of favorites
        return self.favorites

    def addFeedback(self, movieId: int, feedback: int) -> None:
        # Store feedback score (like=5, dislike=1, etc.)
        self.feedbackHistory[movieId] = feedback

    def buildContentVector(self, featureMatrix: pd.DataFrame) -> pd.Series:
        # Make sure there are favorite movies to use
        if not self.favorites:
            raise ValueError("No favorite movies to build content vector.")

        # Get features for the favorite movies only
        fav_features = featureMatrix.loc[self.favorites]

        # Average them to make a user profile vector
        self.contentVector = fav_features.mean(axis=0)
        return self.contentVector

    def updateCollaborativeVector(self, userFactors: pd.DataFrame) -> None:
        # Look up the user's learned collaborative vector
        if self.userId not in userFactors.index:
            raise ValueError(f"User {self.userId} not found in collaborative model factors.")

        self.collabVector = userFactors.loc[self.userId]

    def getProfileSummary(self) -> dict:
        # Summarize what's in the user profile
        return {
            "userId": self.userId,
            "favorites": self.favorites,
            "feedbackHistory": self.feedbackHistory,
            "contentVectorShape": None if self.contentVector is None else self.contentVector.shape,
            "collabVectorShape": None if self.collabVector is None else self.collabVector.shape
        }
