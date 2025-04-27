from utils.userProfile import UserProfile
import pandas as pd

class UserProfileTester:
    def __init__(self, metadata: pd.DataFrame, featureMatrix: pd.DataFrame):
        self.metadata = metadata  # cleaned_metadata
        self.featureMatrix = featureMatrix  # joined features (cat + tfidf + votes)
        self.userProfile = None

    def run(self):
        print("\n🧑‍💻 Running UserProfileTester...\n")

        # Step 1. Initialize a test user
        self.userProfile = UserProfile(userId=1)
        print(f"✅ Created UserProfile for userId: {self.userProfile.userId}")

        # Step 2. Add favorite movies
        favorite_movie_ids = self.metadata["movie_id"].head(5).tolist()
        self.userProfile.addFavorites(favorite_movie_ids)
        print(f"🎬 Added favorites: {favorite_movie_ids}")

        # Step 3. Build the content vector based on favorites
        self.featureMatrix.index = self.metadata["movie_id"]  # make sure index matches
        content_vector = self.userProfile.buildContentVector(self.featureMatrix)
        print(f"🧠 Content Vector Shape: {content_vector.shape}")

        # Step 4. Add feedback manually
        feedback_examples = {favorite_movie_ids[0]: 1, favorite_movie_ids[1]: 0}
        for movieId, feedback in feedback_examples.items():
            self.userProfile.addFeedback(movieId, feedback)
        print(f"📝 Feedback History: {self.userProfile.feedbackHistory}")

        # Step 5. Profile summary
        summary = self.userProfile.getProfileSummary()
        print("\n📄 Profile Summary:")
        for k, v in summary.items():
            print(f"{k}: {v}")

        return {
            "user_profile": self.userProfile,
            "content_vector": content_vector
        }

if __name__ == "__main__":
    from tests.dataTester import DataTester

    # First run the DataTester to get cleaned metadata and feature matrices
    data_outputs = DataTester().run()

    # Build a combined feature matrix (cat + votes)
    featureMatrix = data_outputs["cat_features"].join(data_outputs["votes"])
    featureMatrix.index = data_outputs["metadata"]["movie_id"]  # align index!

    # Run UserProfileTester
    tester = UserProfileTester(
        metadata=data_outputs["metadata"],
        featureMatrix=featureMatrix
    )
    tester.run()
