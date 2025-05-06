import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.helpers import normalizeVectors
from utils.omdbFetcher import OmdbFetcher

# Load metadata from OMDb or from cached file
class IMDbLoader:
    def __init__(self, linksPath: str, apiKey: str, cachePath: str = "ml-100k/omdb_metadata.csv"):
        self.linksPath = linksPath
        self.apiKey = apiKey
        self.cachePath = cachePath
        self.metadataDF = None
        self.fetcher = OmdbFetcher(apiKey, cachePath)

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

    def preprocessMetadata(self) -> pd.DataFrame:
        # Drop rows with missing titles or plots, fill missing ratings
        return self.metadataDF.dropna(subset=["title", "overview"]).fillna({"voteAverage": 0})

# Load MovieLens ratings
class MovieLensLoader:
    def __init__(self, ratingsPath: str):
        self.ratingsPath = ratingsPath

    def loadRatings(self) -> pd.DataFrame:
        return pd.read_csv(self.ratingsPath)

# Build feature vectors from metadata (genres, actors, etc.)
class MetadataPreprocessor:
    def __init__(self, metadataDF: pd.DataFrame):
        self.metadataDF = metadataDF

    def encodeCategoricalFeatures(self) -> pd.DataFrame:
        mlb = MultiLabelBinarizer()

        # Encode genres
        genres = mlb.fit_transform(self.metadataDF["genres"])
        genresDF = pd.DataFrame(genres, columns=[f"genre_{g}" for g in mlb.classes_])

        # Encode directors
        directors = mlb.fit_transform(self.metadataDF["directors"])
        directorsDF = pd.DataFrame(directors, columns=[f"director_{d}" for d in mlb.classes_])

        # Encode actors
        actors = mlb.fit_transform(self.metadataDF["actors"])
        actorsDF = pd.DataFrame(actors, columns=[f"actor_{a}" for a in mlb.classes_])

        return pd.concat([genresDF, directorsDF, actorsDF], axis=1)

    def applyTfidfToPlots(self) -> pd.DataFrame:
        # Convert movie plots into TF-IDF matrix
        tfidf = TfidfVectorizer(max_features=100, stop_words="english")
        matrix = tfidf.fit_transform(self.metadataDF["overview"].fillna(""))
        return pd.DataFrame(matrix.toarray(), columns=tfidf.get_feature_names_out())

    def normalizeVoteAverage(self) -> pd.DataFrame:
        # Normalize average IMDb vote scores to range [0, 1]
        voteAvg = self.metadataDF[["voteAverage"]]
        voteAvgScaled = normalizeVectors(voteAvg)
        voteAvgScaled.columns = ["voteAvgScaled"]
        return voteAvgScaled

# Convert MovieLens ratings into binary (like/dislike)
class RatingsPreprocessor:
    def __init__(self, ratingsDF: pd.DataFrame):
        self.ratingsDF = ratingsDF

    def binarizeRatings(self, threshold: float = 3.5) -> pd.DataFrame:
        # Label ratings >= threshold as 1 (like), else 0 (dislike)
        self.ratingsDF["binaryRating"] = (self.ratingsDF["rating"] >= threshold).astype(int)
        return self.ratingsDF