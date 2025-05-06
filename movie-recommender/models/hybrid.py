from typing import List
import pandas as pd
import numpy as np

class HybridRecommender:
     # Use both content-based and collaborative models
    def __init__(self, contentModel, collabModel, alpha: float = 0.5):
        self.contentModel = contentModel
        self.collabModel = collabModel
        self.alpha = alpha  # controls how much weight to give to each model
     # Get content-based scores by dot product with user profile

    def blendScores(self, userId: int, userProfile: pd.Series) -> pd.Series:
        contentScores = pd.Series(
            self.contentModel.featureMatrix @ userProfile,
            index=self.contentModel.featureMatrix.index
        )

        # Get collaborative scores (predicted ratings)
        collabScores = {}
        for movieId in self.contentModel.featureMatrix.index:
            if movieId in self.collabModel.movieIdMapping:
                collabScores[movieId] = self.collabModel.predictRating(userId, movieId)
            else:
                collabScores[movieId] = 0.0  # default score if no data

        collabScores = pd.Series(collabScores)

        # Normalize both scores to 0–1 range
        contentScores = (contentScores - contentScores.min()) / (contentScores.max() - contentScores.min() + 1e-8)
        collabScores = (collabScores - collabScores.min()) / (collabScores.max() - collabScores.min() + 1e-8)

        # Combine the two using alpha weight
        blended = self.alpha * contentScores + (1 - self.alpha) * collabScores
        return blended
    
     # Return top N movie IDs based on blended scores
    def recommendMovies(self, userId: int, userProfile: pd.Series, topN: int = 10) -> List[int]:    
        blendedScores = self.blendScores(userId, userProfile)
        return blendedScores.sort_values(ascending=False).head(topN).index.tolist()
   
    # Change blend weight (if user has very few ratings)
    def updateAlpha(self, newAlpha: float) -> None:   
        self.alpha = newAlpha
