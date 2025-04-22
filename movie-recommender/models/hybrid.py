from typing import List
import pandas as pd

class HybridRecommender:
    def __init__(self, contentModel, collabModel, alpha: float = 0.5):
        """
        :param contentModel: Instance of ContentBasedRecommender
        :param collabModel: Instance of CollaborativeRecommender
        :param alpha: Weighting factor between content and collaborative (0–1)
        """
        self.contentModel = contentModel
        self.collabModel = collabModel
        self.alpha = alpha  # 0 = all collab, 1 = all content

    # Combine scores from content and collaborative recommenders
    def blendScores(self, userId: int, userProfile: pd.Series) -> pd.Series:
        pass

    # Recommend top-N movies using blended scores
    def recommendMovies(self, userId: int, userProfile: pd.Series, topN: int = 10) -> List[int]:
        pass

    # Adjust hybrid strategy (increase content weight in cold-start)
    def updateAlpha(self, newAlpha: float) -> None:
        pass
