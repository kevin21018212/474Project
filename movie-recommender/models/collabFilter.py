import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from utils.omdbFetcher import OmdbFetcher
from typing import List

class CollaborativeRecommender:
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
        uIdx = self.userIdMapping[userId]
        mIdx = self.movieIdMapping[movieId]
        # Calculate the dot product of user and movie factors to get the predicted rating
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
        # Get the index in matrix
        uIdx = self.userIdMapping[userId]
        mIdx = self.movieIdMapping[movieId]
        
        # Calculate the error 
        currentVector = self.userFactors[uIdx]
        movieVector = self.movieFactors[mIdx]
        error = feedback - np.dot(currentVector, movieVector)

        # Update using Stochastic Gradient Descent (SGD)
        learningRate = 0.1  #controls the update size
        self.userFactors[uIdx] += learningRate * error * movieVector

    def getMovieTitle(self, movieId: int, fetcher: OmdbFetcher) -> str:
        # Look up the movie title in the metadata
        match = self.metadataDF[self.metadataDF["movieId"] == movieId]
        
        if not match.empty:
            return match["title"].values[0]  

        # If movieId is not found, fetch imdbId, use OMDb to fetch stitle
        imdbId = self.linksDF[self.linksDF["movieId"] == movieId]["imdbId"].values
        if not imdbId:
            return "Unknown Title" 
        
        imdbId = imdbId[0]  #first imdbId
        
        # Fetch title from OMDb using the fetched imdbId
        fetched = fetcher.fetchMovie(movieId, imdbId)
        if fetched and 'title' in fetched:
            # Append the fetched movie data to metadataDF and save the updated metadata
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

            # Save the updated metadata to the CSV file
            self.metadataDF.to_csv("ml-100k/omdb_metadata.csv", index=False)
            fetcher.saveCache()  # Update the cache file with the new movie data
            
            return fetched['title']  

        return "Unknown Title"  