from utils.userProfile import UserProfile
import pandas as pd

class UserProfileTester:
    def __init__(self, metadata: pd.DataFrame, featureMatrix: pd.DataFrame):
        self.metadata = metadata
        self.featureMatrix = featureMatrix
        self.userProfile = None

    def run(self):
        # Initialize user
        self.userProfile = UserProfile(userId=1)

        # Add favorite movies (first 5)
        favoriteMovieIds = self.metadata["movieId"].head(5).tolist()
        self.userProfile.addFavorites(favoriteMovieIds)

        # Display favorite movies with details
        print("\n🎬 Favorite Movies:")
        for movieId in favoriteMovieIds:
            movie = self.metadata[self.metadata["movieId"] == movieId].iloc[0]
            print(f" - {movie['title']} ({movie['genres']}) by {movie['directors']}")

        # Build content vector
        self.featureMatrix.index = self.metadata["movieId"]
        contentVector = self.userProfile.buildContentVector(self.featureMatrix)
        print(f"\n🧠 Built content vector with shape: {contentVector.shape}")

        # Add example feedback manually
        feedbackExamples = {favoriteMovieIds[0]: 1, favoriteMovieIds[1]: 0}
        for movieId, feedback in feedbackExamples.items():
            self.userProfile.addFeedback(movieId, feedback)

        # Show feedback history nicely
        print("\n📝 Feedback History:")
        for movieId, feedback in self.userProfile.feedbackHistory.items():
            movieTitle = self.metadata[self.metadata["movieId"] == movieId]["title"].values[0]
            label = "Liked" if feedback == 1 else "Disliked"
            print(f" - {movieTitle}: {label}")

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
