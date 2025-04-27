from typing import List
import numpy as np
import pandas as pd
from models.collabFilter import CollaborativeFilter
from utils.userProfile import UserProfile
from tests.dataTester import DataTester


class CollaborativeFilterTester:
    def __init__(self, userProfile: UserProfile, ratingsDF: pd.DataFrame, metadataDF: pd.DataFrame):
        self.userProfile = userProfile
        self.ratingsDF = ratingsDF
        self.metadataDF = metadataDF
        self.collabModel = None

    def run(self):
        print("\n🚀 Running CollaborativeFilterTester...\n")

        # Initialize and train collaborative model
        self.collabModel = CollaborativeFilter(numFactors=30)
        self.collabModel.trainModel(self.ratingsDF)
        print("✅ Trained collaborative model.")

        # Recommend top-5 movies for the given user
        topMovies = self.collabModel.recommendMovies(userId=self.userProfile.userId, topN=5)
        print(f"\n🎬 Top-5 Recommended Movies for user {self.userProfile.userId}:")
        for movieId in topMovies:
            title = self._getMovieTitle(movieId)
            print(f" - {title} ({movieId})")

        # Simulate feedback (user likes the top recommended movie)
        print("\n🛠 Updating user vector based on feedback...")
        feedbackMovieId = topMovies[0]
        self.collabModel.updateUserVector(self.userProfile.userId, feedbackMovieId, feedback=1)

        # Predict again after feedback
        updatedPrediction = self.collabModel.predictRating(self.userProfile.userId, feedbackMovieId)
        updatedTitle = self._getMovieTitle(feedbackMovieId)
        print(f"✅ Updated predicted rating for {updatedTitle}: {updatedPrediction:.3f}")

        return {
            "collabModel": self.collabModel,
            "topMovies": topMovies
        }

    def _getMovieTitle(self, movieId: int) -> str:
        match = self.metadataDF[self.metadataDF["movieId"] == movieId]
        if match.empty:
            return "Unknown Title"
        return match["title"].values[0]


if __name__ == "__main__":
    # Load the data
    dataOutputs = DataTester().run()
    ratingsDF = dataOutputs["binaryRatings"]
    metadataDF = dataOutputs["metadata"]

    # Build the user profile 
    from tests.userProfileTester import UserProfileTester
    userProfileOutputs = UserProfileTester(metadata=metadataDF, featureMatrix=dataOutputs["featureMatrix"]).run()
    userProfile = userProfileOutputs["userProfile"]

    # Run collaborative filter tester
    tester = CollaborativeFilterTester(userProfile=userProfile, ratingsDF=ratingsDF, metadataDF=metadataDF)
    tester.run()
