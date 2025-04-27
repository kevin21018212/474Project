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
        print(f"✅ Created UserProfile for userId: {self.userProfile.userId}")

        # Add favorite movies
        favorite_movie_ids = self.metadata["movie_id"].head(5).tolist()
        self.userProfile.addFavorites(favorite_movie_ids)
        print(f" Added favorites: {favorite_movie_ids}")

        # Build content vector
        self.featureMatrix.index = self.metadata["movie_id"]
        content_vector = self.userProfile.buildContentVector(self.featureMatrix)
        print(f" Built content vector of shape: {content_vector.shape}")

        # Add feedback
        feedback_examples = {favorite_movie_ids[0]: 1, favorite_movie_ids[1]: 0}
        for movieId, feedback in feedback_examples.items():
            self.userProfile.addFeedback(movieId, feedback)
        print(f" Feedback history recorded: {self.userProfile.feedbackHistory}")

        # Print basic profile summary
        summary = self.userProfile.getProfileSummary()
        print("\n Profile Summary:")
        print(summary)

        return {
            "user_profile": self.userProfile,
            "content_vector": content_vector
        }

if __name__ == "__main__":
    from tests.dataTester import DataTester

    data_outputs = DataTester().run()
    featureMatrix = data_outputs["featureMatrix"]
    metadata = data_outputs["metadata"]

    tester = UserProfileTester(
        metadata=metadata,
        featureMatrix=featureMatrix
    )
    tester.run()
