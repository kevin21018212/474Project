import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from utils.omdbFetcher import OmdbFetcher
from typing import List

class CollaborativeFilter:
    def __init__(self, numFactors: int = 30, metadataDF: pd.DataFrame = None):
        self.numFactors = numFactors
        self.metadataDF = metadataDF  # Movie metadata (movies, titles, etc.)
        self.linksDF = pd.read_csv("ml-100k/links.csv")  # Mapping of movieId to imdbId 

    # Create a matrix of users and movies based on ratings
    def trainModel(self, ratingsDF: pd.DataFrame) -> None:
       
        # Rows: users, Columns: movies, Values: ratings
        self.interactionMatrix = ratingsDF.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

        # Create a mapping from userId/movieId to matrix indices
        self.userIdMapping = {uid: idx for idx, uid in enumerate(self.interactionMatrix.index)}
        self.movieIdMapping = {mid: idx for idx, mid in enumerate(self.interactionMatrix.columns)}
        
        #Apply Singular Value Decomposition (SVD) to reduce dimensions
        svd = TruncatedSVD(n_components=self.numFactors, random_state=42)
        reducedMatrix = svd.fit_transform(self.interactionMatrix)

        # Store the reduced matrices for users and movies
        self.userFactors = reducedMatrix  # Matrix with user factor representations
        self.movieFactors = svd.components_.T  # Matrix with movie factor representations
    
    # Predict the rating for a given user and movie
    def predictRating(self, userId: int, movieId: int) -> float:
        # Handle cold-start users
        if userId not in self.userIdMapping:
            newIndex = len(self.userFactors)
            self.userIdMapping[userId] = newIndex
            newVec = np.zeros(self.userFactors.shape[1])
            self.userFactors = np.vstack([self.userFactors, newVec])

        if movieId not in self.movieIdMapping:
            return 0.0

        uIdx = self.userIdMapping[userId]
        mIdx = self.movieIdMapping[movieId]
        return np.dot(self.userFactors[uIdx], self.movieFactors[mIdx])


    def recommendMovies(self, userId: int, topN: int = 10) -> List[int]:
        # Get the user’s factor vector and compute similarity with all movies
        uIdx = self.userIdMapping[userId]
        scores = self.userFactors[uIdx] @ self.movieFactors.T  # Dot product to calculate movie scores

        # Get the top N movie indices based on the scores
        topIndices = np.argsort(scores)[-topN:][::-1]
        
        # Convert the movie indices back to movieIds using the movieId mapping
        movieIdxToId = {idx: mid for mid, idx in self.movieIdMapping.items()}
        topMovieIds = [movieIdxToId[i] for i in topIndices]
        return topMovieIds

    # Update the user’s vector based on their feedback (like/dislike)
    def updateUserVector(self, userId: int, movieId: int, feedback: int) -> None:
        if userId not in self.userIdMapping:
            self.userIdMapping[userId] = len(self.userFactors)
            newUserVector = np.zeros(self.userFactors.shape[1])
            self.userFactors = np.vstack([self.userFactors, newUserVector])

        uIdx = self.userIdMapping[userId]
        mIdx = self.movieIdMapping.get(movieId)
        if mIdx is None:
            return

        currentVector = self.userFactors[uIdx]
        movieVector = self.movieFactors[mIdx]
        error = feedback - np.dot(currentVector, movieVector)
        self.userFactors[uIdx] += 0.1 * error * movieVector