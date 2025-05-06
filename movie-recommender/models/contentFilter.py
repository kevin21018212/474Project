import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from utils.helpers import normalizeVectors

class ContentBasedFilter:
    def __init__(self, metadataDF: pd.DataFrame):
        # Movie metadata (title, genres, plot, etc.)
        self.metadataDF = metadataDF
        self.featureMatrix = None         # Matrix of all movie features
        self.movieIdToIndex = {}          # Mapping from movie ID to row index

    # Turn genres, directors, actors, plots, and ratings into machine-readable numbers
    def buildFeatureMatrix(self) -> None:
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.feature_extraction.text import TfidfVectorizer

        mlb = MultiLabelBinarizer()

        # One-hot encode genres
        genres = mlb.fit_transform(self.metadataDF["genres"])
        genresDF = pd.DataFrame(genres, columns=[f"genre_{g}" for g in mlb.classes_])

        # One-hot encode directors
        directors = mlb.fit_transform(self.metadataDF["directors"])
        directorsDF = pd.DataFrame(directors, columns=[f"director_{d}" for d in mlb.classes_])

        # One-hot encode actors
        actors = mlb.fit_transform(self.metadataDF["actors"])
        actorsDF = pd.DataFrame(actors, columns=[f"actor_{a}" for a in mlb.classes_])

        # Convert plot text to TF-IDF features
        tfidf = TfidfVectorizer(max_features=100, stop_words="english")
        plotTFIDF = tfidf.fit_transform(self.metadataDF["overview"].fillna(""))
        plotDF = pd.DataFrame(plotTFIDF.toarray(), columns=tfidf.get_feature_names_out())

        # Normalize IMDb rating (0–1)
        voteAvg = normalizeVectors(self.metadataDF[["voteAverage"]])
        voteAvg.columns = ["voteAvgScaled"]

        # Combine all features into one matrix
        self.featureMatrix = pd.concat([genresDF, directorsDF, actorsDF, plotDF, voteAvg], axis=1)
        self.featureMatrix.index = self.metadataDF["movieId"]
        self.movieIdToIndex = {mid: idx for idx, mid in enumerate(self.metadataDF["movieId"])}

    # Average the feature vectors of all the user's favorite movies
    def buildUserProfile(self, favoriteMovieIds: List[int]) -> pd.Series:
        validIds = [mid for mid in favoriteMovieIds if mid in self.featureMatrix.index]
        if not validIds:
            # If no valid movies, return zero vector
            return pd.Series(np.zeros(self.featureMatrix.shape[1]), index=self.featureMatrix.columns)

        return self.featureMatrix.loc[validIds].mean()
   
     # Compare the user profile to every movie using cosine similarity
    def recommendMovies(self, userProfile: pd.Series, topN: int = 10) -> List[int]:   
        sims = cosine_similarity([userProfile], self.featureMatrix)[0]
        # Return top N most similar movie IDs
        topIndices = np.argsort(sims)[-topN:][::-1]
        return self.featureMatrix.index[topIndices].tolist()

    # Fine-tune user profile based on feedback (1 = like, 0 = dislike)
    def updateUserProfile(self, userProfile: pd.Series, movieId: int, feedback: int) -> pd.Series:
        if movieId not in self.featureMatrix.index:
            return userProfile

        movieVector = self.featureMatrix.loc[movieId]
        alpha = 0.1  # how fast the profile updates

        # Move user profile closer or further from the movie
        if feedback == 1:
            userProfile += alpha * (movieVector - userProfile)
        else:
            userProfile -= alpha * (movieVector - userProfile)

        return userProfile
