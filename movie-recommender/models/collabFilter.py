from typing import List
import numpy as np
import pandas as pd

class CollaborativeFilter:
    def __init__(self, numFactors: int = 30, regularization: float = 0.1, learningRate: float = 0.01):
        self.numFactors = numFactors
        self.regularization = regularization
        self.learningRate = learningRate
        self.userFactors = None
        self.movieFactors = None

    # Train collaborative filtering model using matrix factorization
    def trainModel(self, ratingsDF: pd.DataFrame) -> None:
        pass

    # Predict rating for a specific user and movie
    def predictRating(self, userId: int, movieId: int) -> float:
        pass

    # Generate top-N movie recommendations for a given user
    def recommendMovies(self, userId: int, topN: int = 10) -> List[int]:
        pass

    # Update user preferences based on new feedback (like/dislike)
    def updateUserVector(self, userId: int, movieId: int, feedback: int) -> None:
        pass
