from typing import List
import numpy as np
import pandas as pd
from models.collabFilter import CollaborativeFilter
from models.contentFilter import ContentBasedFilter
from models.hybrid import HybridRecommender
from utils.userProfile import UserProfile
from tests.dataTester import DataTester
from utils.omdbFetcher import OmdbFetcher

class CollaborativeFilterTester:
    def __init__(self, userProfile: UserProfile, ratingsDF: pd.DataFrame, metadataDF: pd.DataFrame, fetcher: OmdbFetcher):
        self.userProfile = userProfile
        self.ratingsDF = ratingsDF
        self.metadataDF = metadataDF
        self.fetcher = fetcher
        self.collabModel = CollaborativeFilter(numFactors=30, metadataDF=metadataDF)

    def run(self):
        print("\n Running CollaborativeFilterTester...\n")
        self.collabModel.trainModel(self.ratingsDF)
        print(" Trained collaborative model.")

        topMovies = self.collabModel.recommendMovies(userId=self.userProfile.userId, topN=5)
        print(f"\n Top-5 Recommended Movies for user {self.userProfile.userId}:")
        for movieId in topMovies:
            title = self._getMovieTitle(movieId)
            score = self.collabModel.predictRating(self.userProfile.userId, movieId)
            print(f" - {title} ({movieId}) | Predicted rating: {score:.3f}")
        print("\n Updating user vector based on feedback...")
        feedbackMovieId = topMovies[0]
        self.collabModel.updateUserVector(self.userProfile.userId, feedbackMovieId, feedback=1)
        updatedPrediction = self.collabModel.predictRating(self.userProfile.userId, feedbackMovieId)
        updatedTitle = self._getMovieTitle(feedbackMovieId)
        print(f" Updated predicted rating for {updatedTitle}: {updatedPrediction:.3f}")

        newTopMovies = self.collabModel.recommendMovies(userId=self.userProfile.userId, topN=5)
        print("\n New Top-5 Recommendations After Feedback:")
        for movieId in newTopMovies:
            title = self._getMovieTitle(movieId)
            score = self.collabModel.predictRating(self.userProfile.userId, movieId)
            print(f" - {title} ({movieId}) | Score: {score:.3f}")

        return {
            "collabModel": self.collabModel,
            "topMovies_before": topMovies,
            "topMovies_after": newTopMovies
        }

    def _getMovieTitle(self, movieId: int) -> str:
        row = self.metadataDF[self.metadataDF["movieId"] == movieId]
        return row["title"].values[0] if not row.empty else "Unknown"

class ContentBasedFilterTester:
    def __init__(self, userProfile: UserProfile, contentModel: ContentBasedFilter, metadataDF: pd.DataFrame):
        self.userProfile = userProfile
        self.contentModel = contentModel
        self.metadataDF = metadataDF

    def run(self):
        print("\n Running ContentBasedFilterTester...\n")
        profileVec = self.contentModel.buildUserProfile(self.userProfile.favorites)
        sims = self.contentModel.featureMatrix @ profileVec
        ranked = sims.sort_values(ascending=False).head(5)

        print(f"\n Top-5 Content-Based Recommendations:")
        for movieId, score in ranked.items():
            title = self._getMovieTitle(movieId)
            print(f" - {title} ({movieId}) | Similarity: {score:.3f}")

    def _getMovieTitle(self, movieId: int) -> str:
        row = self.metadataDF[self.metadataDF["movieId"] == movieId]
        return row["title"].values[0] if not row.empty else "Unknown"

class HybridFilterTester:
    def __init__(self, userProfile: UserProfile, hybridModel: HybridRecommender, metadataDF: pd.DataFrame):
        self.userProfile = userProfile
        self.hybridModel = hybridModel
        self.metadataDF = metadataDF

    def run(self):
        print("\n Running HybridFilterTester...\n")
        profileVec = self.hybridModel.contentModel.buildUserProfile(self.userProfile.favorites)
        hybridScores = self.hybridModel.blendScores(self.userProfile.userId, profileVec)
        topMovies = hybridScores.sort_values(ascending=False).head(5)

        print(f"\nTop-5 Hybrid Recommendations:")
        for movieId, score in topMovies.items():
            title = self._getMovieTitle(movieId)
            print(f" - {title} ({movieId}) | Hybrid Score: {score:.3f}")

    def _getMovieTitle(self, movieId: int) -> str:
        row = self.metadataDF[self.metadataDF["movieId"] == movieId]
        return row["title"].values[0] if not row.empty else "Unknown"

if __name__ == "__main__":
    dataOutputs = DataTester().run()
    ratingsDF = dataOutputs["binaryRatings"]
    metadataDF = dataOutputs["metadata"]
    featureMatrix = dataOutputs["featureMatrix"]

    from tests.userProfileTester import UserProfileTester
    userProfileOutputs = UserProfileTester(metadata=metadataDF, featureMatrix=featureMatrix).run()
    userProfile = userProfileOutputs["userProfile"]

    fetcher = OmdbFetcher(apiKey="766c1b0d")

    # Run collaborative filter test
    collabTester = CollaborativeFilterTester(userProfile, ratingsDF, metadataDF, fetcher)
    collabTester.run()

    # Run content-based test
    contentModel = ContentBasedFilter(metadataDF)
    contentModel.featureMatrix = featureMatrix
    contentModel.movieIdToIndex = {mid: idx for idx, mid in enumerate(metadataDF["movieId"])}
    contentTester = ContentBasedFilterTester(userProfile, contentModel, metadataDF)
    contentTester.run()

    # Run hybrid test
    collabModel = CollaborativeFilter(numFactors=30, metadataDF=metadataDF)
    collabModel.trainModel(ratingsDF)
    hybridModel = HybridRecommender(contentModel, collabModel, alpha=0.5)
    hybridTester = HybridFilterTester(userProfile, hybridModel, metadataDF)
    hybridTester.run()
