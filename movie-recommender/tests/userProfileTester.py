from utils.userProfile import UserProfile
import pandas as pd
import random

class UserProfileTester:
    def __init__(self, metadata: pd.DataFrame, featureMatrix: pd.DataFrame):
        self.metadata = metadata
        self.featureMatrix = featureMatrix
        self.userProfile = None

    def run(self):
        print("\n🚀 Running User Profile Tests: \n")

        # Initialize user
        self.userProfile = UserProfile(userId=1)

        # Add favorite movies (increase to first 20 movies)
        favoriteMovieIds = self.metadata["movieId"].head(20).tolist()  # Increased from 5 to 20
        self.userProfile.addFavorites(favoriteMovieIds)

        # Build content vector
        self.featureMatrix.index = self.metadata["movieId"]
        contentVector = self.userProfile.buildContentVector(self.featureMatrix)
        print(f"\n Built content vector with shape: {contentVector.shape}")

        # Add more feedback for the user (increase from 2 to 10 feedbacks)
        feedbackExamples = {movieId: random.choice([1, 0]) for movieId in favoriteMovieIds[:10]}  # Give random feedback for the first 10 movies
        for movieId, feedback in feedbackExamples.items():
            self.userProfile.addFeedback(movieId, feedback)

        print("\n Feedback History:")
        for movieId, feedback in self.userProfile.feedbackHistory.items():
            movie = self.metadata[self.metadata["movieId"] == movieId].iloc[0]
            movieTitle = movie["title"]
            genres = movie["genres"]
            label = "Liked" if feedback == 1 else "Disliked"
            print(f" - {movieTitle}: {label}, Genres: {genres}")

        # Print profile summary
        summary = self.userProfile.getProfileSummary()
        print("\n📄 Profile Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")

        return {
            "userProfile": self.userProfile,
            "contentVector": contentVector
        }

if __name__ == "__main__":
    from tests.dataTester import DataTester

    # Load data
    dataOutputs = DataTester().run()
    featureMatrix = dataOutputs["featureMatrix"]
    metadata = dataOutputs["metadata"]

    # Run user profile tester
    tester = UserProfileTester(
        metadata=metadata,
        featureMatrix=featureMatrix
    )
    tester.run()
