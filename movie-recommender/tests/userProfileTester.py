# ✅ File 2: tests/userProfileTester.py
from utils.userProfile import UserProfile
from tests.dataTester import DataTester
import pandas as pd
import random

class UserProfileTester:
    def __init__(self, metadata: pd.DataFrame, featureMatrix: pd.DataFrame):
        self.metadata = metadata
        self.featureMatrix = featureMatrix
        self.userProfile = None

    def run(self):
        print("\nRunning User Profile Tester\n")
        self.userProfile = UserProfile(userId=1)
        fav_ids = self.metadata["movieId"].head(10).tolist()
        self.userProfile.addFavorites(fav_ids)

        print(f" Added {len(fav_ids)} favorites for user {self.userProfile.userId}:")
     

        self.featureMatrix.index = self.metadata["movieId"]
        profileVec = self.userProfile.buildContentVector(self.featureMatrix)

        print(f"\n Built user content vector. Shape: {profileVec.shape}")
        print(f" Content vector: {profileVec[:10]}")

        print("\n Feedback on favorites:")
        for movieId in fav_ids[:5]:
            feedback = random.choice([0, 1])
            self.userProfile.addFeedback(movieId, feedback)
            title = self.metadata.loc[self.metadata.movieId == movieId, "title"].values[0]
            print(f"   - {'Like' if feedback else ' Dislike'} on '{title}' (movieId={movieId})")

        print("\n Profile Summary:")
        print(self.userProfile.getProfileSummary())

        print("\nCos similarity(top 5 favorite movies):")
        from sklearn.metrics.pairwise import cosine_similarity
        fav_matrix = self.featureMatrix.loc[fav_ids]
        similarities = cosine_similarity([profileVec], fav_matrix)[0]
        for mid, sim in zip(fav_ids[:5], similarities[:5]):
            title = self.metadata.loc[self.metadata.movieId == mid, "title"].values[0]
            print(f"   - {title} (movieId={mid}): {sim:.3f}")

        return {
            "userProfile": self.userProfile,
            "contentVector": profileVec
        }

    

if __name__ == "__main__":
    data = DataTester().run()
    tester = UserProfileTester(data["metadata"], data["featureMatrix"])
    tester.run()

