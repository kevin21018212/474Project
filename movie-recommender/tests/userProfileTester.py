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
        favorite_movie_ids = self.metadata["movie_id"].head(5).tolist()
        self.userProfile.addFavorites(favorite_movie_ids)

        # Display favorite movies with details
        print("\n Favorite Movies:")
        for mid in favorite_movie_ids:
            movie = self.metadata[self.metadata["movie_id"] == mid].iloc[0]
            print(f" - {movie['title']} ({movie['genres']}) by {movie['directors']}")

        # Build content vector
        self.featureMatrix.index = self.metadata["movie_id"]
        content_vector = self.userProfile.buildContentVector(self.featureMatrix)
        print(f"\n Built content vector with shape: {content_vector.shape}")

        # Add example feedback manually
        feedback_examples = {favorite_movie_ids[0]: 1, favorite_movie_ids[1]: 0}
        for movieId, feedback in feedback_examples.items():
            self.userProfile.addFeedback(movieId, feedback)

        # Show feedback history in a friendly way
        print("\n Feedback History:")
        for movieId, feedback in self.userProfile.feedbackHistory.items():
            movie_title = self.metadata[self.metadata["movie_id"] == movieId]["title"].values[0]
            label = "Liked" if feedback == 1 else " Disliked"
            print(f" - {movie_title}: {label}")

        # Profile summary
        summary = self.userProfile.getProfileSummary()
        print("\n Profile Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")

        return {
            "user_profile": self.userProfile,
            "content_vector": content_vector
        }

if __name__ == "__main__":
    from tests.dataTester import DataTester

    # Load data
    data_outputs = DataTester().run()
    featureMatrix = data_outputs["featureMatrix"]
    metadata = data_outputs["metadata"]

    # Run user profile tester
    tester = UserProfileTester(
        metadata=metadata,
        featureMatrix=featureMatrix
    )
    tester.run()
