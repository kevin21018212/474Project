import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from utils.helpers import normalizeVectors

class ContentBasedFilter:
    def __init__(self, metadataDF: pd.DataFrame):
        self.metadataDF = metadataDF
        self.featureMatrix = None
        self.movieIdToIndex = {}

    # Build feature matrix from metadata (genres, directors, actors, plot, voteAvg)
    def buildFeatureMatrix(self) -> None:
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.feature_extraction.text import TfidfVectorizer

        mlb = MultiLabelBinarizer()
        genres = mlb.fit_transform(self.metadataDF["genres"])
        genresDF = pd.DataFrame(genres, columns=[f"genre_{g}" for g in mlb.classes_])

        directors = mlb.fit_transform(self.metadataDF["directors"])
        directorsDF = pd.DataFrame(directors, columns=[f"director_{d}" for d in mlb.classes_])

        actors = mlb.fit_transform(self.metadataDF["actors"])
        actorsDF = pd.DataFrame(actors, columns=[f"actor_{a}" for a in mlb.classes_])

        tfidf = TfidfVectorizer(max_features=100, stop_words="english")
        plotTFIDF = tfidf.fit_transform(self.metadataDF["overview"].fillna(""))
        plotDF = pd.DataFrame(plotTFIDF.toarray(), columns=tfidf.get_feature_names_out())

        voteAvg = normalizeVectors(self.metadataDF[["voteAverage"]])
        voteAvg.columns = ["voteAvgScaled"]

        self.featureMatrix = pd.concat([genresDF, directorsDF, actorsDF, plotDF, voteAvg], axis=1)
        self.featureMatrix.index = self.metadataDF["movieId"]
        self.movieIdToIndex = {mid: idx for idx, mid in enumerate(self.metadataDF["movieId"])}

    # Average the vectors of favorite movies to form a user profile
    def buildUserProfile(self, favoriteMovieIds: List[int]) -> pd.Series:
        validIds = [mid for mid in favoriteMovieIds if mid in self.featureMatrix.index]
        if not validIds:
            return pd.Series(np.zeros(self.featureMatrix.shape[1]), index=self.featureMatrix.columns)
        return self.featureMatrix.loc[validIds].mean()

    # Recommend movies by comparing user profile to all movies
    def recommendMovies(self, userProfile: pd.Series, topN: int = 10) -> List[int]:
        sims = cosine_similarity([userProfile], self.featureMatrix)[0]
        topIndices = np.argsort(sims)[-topN:][::-1]
        return self.featureMatrix.index[topIndices].tolist()

    # Update user profile with new feedback (like/dislike)
    def updateUserProfile(self, userProfile: pd.Series, movieId: int, feedback: int) -> pd.Series:
        if movieId not in self.featureMatrix.index:
            return userProfile

        movieVector = self.featureMatrix.loc[movieId]
        alpha = 0.1  # learning rate
        if feedback == 1:
            userProfile += alpha * (movieVector - userProfile)
        else:
            userProfile -= alpha * (movieVector - userProfile)
        return userProfile