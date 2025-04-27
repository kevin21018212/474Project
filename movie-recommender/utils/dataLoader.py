import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.helpers import normalizeVectors
from utils.omdbFetcher import OmdbFetcher

import pandas as pd
from utils.helpers import normalizeVectors
from utils.omdbFetcher import OmdbFetcher
import os

class IMDbLoader:
    def __init__(self, linksPath: str, apiKey: str, cachePath: str = "ml-100k/omdb_metadata.csv"):
        self.linksPath = linksPath
        self.apiKey = apiKey
        self.cachePath = cachePath
        self.metadataDF = None
        self.fetcher = OmdbFetcher(apiKey, cachePath)

    # Fetch metadata from cache or OMDb using OmdbFetcher
    def loadMetadata(self, limit=None) -> pd.DataFrame:
        if os.path.exists(self.cachePath):
            self.metadataDF = pd.read_csv(self.cachePath)
            print(" Loaded cached OMDb metadata.")
        else:
            print(" No cache found. Fetching from OMDb API...")
            links = pd.read_csv(self.linksPath)
            if limit:
                links = links.head(limit)

            records = []
            for _, row in links.iterrows():
                movieId = row["movieId"]
                imdbId = row["imdbId"]
                movieData = self.fetcher.fetchMovie(movieId, imdbId)
                if movieData:
                    records.append(movieData)

            self.metadataDF = pd.DataFrame(records)
            self.metadataDF.to_csv(self.cachePath, index=False)

        return self.metadataDF

    # Clean metadata (drop missing titles/plots)
    def preprocessMetadata(self) -> pd.DataFrame:
        return self.metadataDF.dropna(subset=["title", "overview"]).fillna({"voteAverage": 0})
#Load user ratings
class MovieLensLoader:
    def __init__(self, ratingsPath: str):
        self.ratingsPath = ratingsPath

    # Load MovieLens ratings
    def loadRatings(self) -> pd.DataFrame:
        return pd.read_csv(self.ratingsPath)

# Build movie feature matrices
class MetadataPreprocessor:
    def __init__(self, metadataDF: pd.DataFrame):
        self.metadataDF = metadataDF

    # One-hot encode genres, directors, and actors
    def encodeCategoricalFeatures(self) -> pd.DataFrame:
        mlb = MultiLabelBinarizer()

        genres = mlb.fit_transform(self.metadataDF["genres"])
        genresDF = pd.DataFrame(genres, columns=[f"genre_{g}" for g in mlb.classes_])

        directors = mlb.fit_transform(self.metadataDF["directors"])
        directorsDF = pd.DataFrame(directors, columns=[f"director_{d}" for d in mlb.classes_])

        actors = mlb.fit_transform(self.metadataDF["actors"])
        actorsDF = pd.DataFrame(actors, columns=[f"actor_{a}" for a in mlb.classes_])

        return pd.concat([genresDF, directorsDF, actorsDF], axis=1)

    # Convert movie plots (overviews) into TF-IDF vectors
    def applyTfidfToPlots(self) -> pd.DataFrame:
        tfidf = TfidfVectorizer(max_features=100, stop_words="english")
        matrix = tfidf.fit_transform(self.metadataDF["overview"].fillna(""))
        return pd.DataFrame(matrix.toarray(), columns=tfidf.get_feature_names_out())

    def normalizeVoteAverage(self) -> pd.DataFrame:
        voteAvg = self.metadataDF[["voteAverage"]]
        voteAvgScaled = normalizeVectors(voteAvg)
        voteAvgScaled.columns = ["voteAvgScaled"]
        return voteAvgScaled

#Binarize user ratings
class RatingsPreprocessor:
    def __init__(self, ratingsDF: pd.DataFrame):
        self.ratingsDF = ratingsDF

    def binarizeRatings(self, threshold: float = 3.5) -> pd.DataFrame:
        self.ratingsDF["binaryRating"] = (self.ratingsDF["rating"] >= threshold).astype(int)
        return self.ratingsDF
