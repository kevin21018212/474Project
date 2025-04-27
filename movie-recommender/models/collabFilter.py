import pandas as pd
from typing import List
import numpy as np
from sklearn.decomposition import TruncatedSVD
from utils.omdbFetcher import OmdbFetcher

class CollaborativeFilter:
    def __init__(self, numFactors: int = 30, metadataDF: pd.DataFrame = None):
        self.numFactors = numFactors
        self.userFactors = None   # Matrix: each user → vector
        self.movieFactors = None  # Matrix: each movie → vector
        self.userIdMapping = None # userId → index in matrix
        self.movieIdMapping = None # movieId → index in matrix
        self.interactionMatrix = None
        self.metadataDF = metadataDF  # Add metadataDF as an attribute
        self.linksDF = pd.read_csv("ml-100k/links.csv")  # Load links.csv to map movieId to imdbId

    # Build user-movie matrix and train SVD model
    def trainModel(self, ratingsDF: pd.DataFrame) -> None:
        # Build a matrix: users as rows, movies as columns
        self.interactionMatrix = ratingsDF.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

        self.userIdMapping = {uid: idx for idx, uid in enumerate(self.interactionMatrix.index)}
        self.movieIdMapping = {mid: idx for idx, mid in enumerate(self.interactionMatrix.columns)}

        # Perform dimensionality reduction with Truncated SVD
        svd = TruncatedSVD(n_components=self.numFactors, random_state=42)
        reducedMatrix = svd.fit_transform(self.interactionMatrix)

        self.userFactors = reducedMatrix               # (users × factors)
        self.movieFactors = svd.components_.T          # (movies × factors)

    # Predict a user's rating for a movie
    def predictRating(self, userId: int, movieId: int) -> float:
        if not self._validUserMovie(userId, movieId):
            return 0.0
        uIdx = self.userIdMapping[userId]
        mIdx = self.movieIdMapping[movieId]
        return np.dot(self.userFactors[uIdx], self.movieFactors[mIdx])

    # Recommend top N movies for a user
    def recommendMovies(self, userId: int, topN: int = 10) -> List[int]:
        if userId not in self.userIdMapping:
            return []

        uIdx = self.userIdMapping[userId]
        scores = self.userFactors[uIdx] @ self.movieFactors.T

        topIndices = np.argsort(scores)[-topN:][::-1]  # get top-N movie indices
        movieIdxToId = {idx: mid for mid, idx in self.movieIdMapping.items()}
        topMovieIds = [movieIdxToId[i] for i in topIndices]
        return topMovieIds

    # Update user vector manually with new feedback (like/dislike)
    def updateUserVector(self, userId: int, movieId: int, feedback: int) -> None:
        if not self._validUserMovie(userId, movieId):
            return

        uIdx = self.userIdMapping[userId]
        mIdx = self.movieIdMapping[movieId]

        # Treat feedback as +1 (liked) or 0 (disliked)
        currentVector = self.userFactors[uIdx]
        movieVector = self.movieFactors[mIdx]

        learningRate = 0.1
        error = feedback - np.dot(currentVector, movieVector)

        # Simple SGD update for the user vector
        self.userFactors[uIdx] += learningRate * error * movieVector

    # Helper: check if user and movie exist
    def _validUserMovie(self, userId: int, movieId: int) -> bool:
        return userId in self.userIdMapping and movieId in self.movieIdMapping

    # Fetch title using OmdbFetcher if not found in metadata
    def getMovieTitle(self, movieId: int, fetcher: OmdbFetcher) -> str:
        # Check if movieId exists in metadataDF
        match = self.metadataDF[self.metadataDF["movieId"] == movieId]
        
        if not match.empty:
            title = match["title"].values[0]
            return title

        # If imdbId is missing in metadataDF, fetch it from links.csv
        print(f"⚡️ MovieId {movieId} not found in metadataDF. Attempting to fetch imdbId from links.csv...")
        imdbId = self.linksDF[self.linksDF["movieId"] == movieId]["imdbId"].values
        if not imdbId:
            print(f"imdbId not found for movieId {movieId} in links.csv")
            return "Unknown Title"
        
        imdbId = imdbId[0]  # Get the imdbId
        
        # Fetch title using OmdbFetcher
        fetched = fetcher.fetchMovie(movieId, imdbId)
        if fetched and 'title' in fetched:
            # Append to the metadata DataFrame
            new_data = {
                "movieId": movieId,
                "title": fetched['title'],
                "genres": fetched.get("genres", []),
                "directors": fetched.get("directors", []),
                "actors": fetched.get("actors", []),
                "overview": fetched.get("overview", ""),
                "voteAverage": fetched.get("voteAverage", 0),
            }
            new_metadata = pd.DataFrame([new_data])
            self.metadataDF = pd.concat([self.metadataDF, new_metadata], ignore_index=True)

            # Save the updated metadata and cache
            self.metadataDF.to_csv("ml-100k/omdb_metadata.csv", index=False)
            fetcher.saveCache()  # Update the cache file as well
            
            return fetched['title']

        print(f"❌ Title not found for movieId {movieId}")
        return "Unknown Title"
