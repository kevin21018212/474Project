import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from utils.omdbFetcher import OmdbFetcher
from typing import List

class CollaborativeFilter:
    def __init__(self, numFactors: int = 30, metadataDF: pd.DataFrame = None):
        self.numFactors = numFactors  # How many patterns to learn 
        self.metadataDF = metadataDF  # metadata about the movies
        self.linksDF = pd.read_csv("ml-100k/links.csv")  # Map from MovieLens to IMDb 

    # Build a matrix: users as rows, movies as columns, ratings as values
    def trainModel(self, ratingsDF: pd.DataFrame) -> None:
        self.interactionMatrix = ratingsDF.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

        # Save index mappings so we can look up positions later
        self.userIdMapping = {uid: idx for idx, uid in enumerate(self.interactionMatrix.index)}
        self.movieIdMapping = {mid: idx for idx, mid in enumerate(self.interactionMatrix.columns)}

        # Apply matrix factorization using SVD 
        svd = TruncatedSVD(n_components=self.numFactors, random_state=42)
        reducedMatrix = svd.fit_transform(self.interactionMatrix)

        self.userFactors = reducedMatrix                    # One row per user
        self.movieFactors = svd.components_.T               # One row per movie

    #If user is new, give them a zero vector (cold start)
    def predictRating(self, userId: int, movieId: int) -> float:
        if userId not in self.userIdMapping:
            newIndex = len(self.userFactors)
            self.userIdMapping[userId] = newIndex
            newVec = np.zeros(self.userFactors.shape[1])
            self.userFactors = np.vstack([self.userFactors, newVec])

        # If the movie wasn't seen during training, return 0 score
        if movieId not in self.movieIdMapping:
            return 0.0

        uIdx = self.userIdMapping[userId]
        mIdx = self.movieIdMapping[movieId]

        # Predict rating by taking dot product between user and movie vectors
        return np.dot(self.userFactors[uIdx], self.movieFactors[mIdx])

     # Get user vector and compute scores for every movie
    def recommendMovies(self, userId: int, topN: int = 10) -> List[int]:
        uIdx = self.userIdMapping[userId]
        scores = self.userFactors[uIdx] @ self.movieFactors.T

        # Get indices of top N movies with highest scores
        topIndices = np.argsort(scores)[-topN:][::-1]

        # Convert matrix indices back to real movieIds
        movieIdxToId = {idx: mid for mid, idx in self.movieIdMapping.items()}
        return [movieIdxToId[i] for i in topIndices]

     # If new user, initialize their vector
    def updateUserVector(self, userId: int, movieId: int, feedback: int) -> None:
        if userId not in self.userIdMapping:
            self.userIdMapping[userId] = len(self.userFactors)
            newUserVector = np.zeros(self.userFactors.shape[1])
            self.userFactors = np.vstack([self.userFactors, newUserVector])

        uIdx = self.userIdMapping[userId]
        mIdx = self.movieIdMapping.get(movieId)

        # If movie isn't known, skip update
        if mIdx is None:
            return

        # Adjust user vector based on feedback (1 = liked, 0 = disliked)
        currentVector = self.userFactors[uIdx]
        movieVector = self.movieFactors[mIdx]
        error = feedback - np.dot(currentVector, movieVector)

        # Simple gradient update (learn from the error)
        self.userFactors[uIdx] += 0.1 * error * movieVector
