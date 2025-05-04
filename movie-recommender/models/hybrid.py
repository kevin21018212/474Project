from typing import List
import pandas as pd
import numpy as np

class HybridRecommender:
    def __init__(self, contentModel, collabModel, alpha: float = 0.5):
        """
        :param contentModel: Instance of ContentBasedFilter
        :param collabModel: Instance of CollaborativeFilter
        :param alpha: Weighting factor between content and collaborative (0–1)
        """
        self.contentModel = contentModel
        self.collabModel = collabModel
        self.alpha = alpha

    def blendScores(self, userId: int, userProfile: pd.Series) -> pd.Series:
        contentScores = pd.Series(
            self.contentModel.featureMatrix @ userProfile,
            index=self.contentModel.featureMatrix.index
        )

        collabScores = {}
        for movieId in self.contentModel.featureMatrix.index:
            if movieId in self.collabModel.movieIdMapping:
                score = self.collabModel.predictRating(userId, movieId)
                collabScores[movieId] = score
            else:
                collabScores[movieId] = 0.0  # fallback or ignore

        collabScores = pd.Series(collabScores)

        # Normalize both
        contentScores = (contentScores - contentScores.min()) / (contentScores.max() - contentScores.min() + 1e-8)
        collabScores = (collabScores - collabScores.min()) / (collabScores.max() - collabScores.min() + 1e-8)

        blended = self.alpha * contentScores + (1 - self.alpha) * collabScores
        return blended


    # Recommend top-N movieIds
    def recommendMovies(self, userId: int, userProfile: pd.Series, topN: int = 10) -> List[int]:
        blendedScores = self.blendScores(userId, userProfile)
        topMovieIds = blendedScores.sort_values(ascending=False).head(topN).index.tolist()
        return topMovieIds

    # Update alpha (e.g. for cold-start handling)
    def updateAlpha(self, newAlpha: float) -> None:
        self.alpha = newAlpha
