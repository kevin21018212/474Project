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
    def encodeCategoricalFeatures(self):
        if 'genres' not in self.metadataDF.columns:
            return pd.DataFrame()  # Return empty DataFrame if no genres
    
        # Ensure genres are strings
        genres = self.metadataDF['genres'].fillna('').astype(str)
    
    # If genres are pipe-separated (e.g., "Action|Adventure")
        if any('|' in g for g in genres):
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            return pd.DataFrame(
                mlb.fit_transform(genres.str.split('|')),
                columns=mlb.classes_,
                index=self.metadataDF.index
            )
        else:
            # Handle single-genre case
            return pd.get_dummies(genres, prefix='genre')

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
    def __init__(self, ratings_df):
        # Ensure required columns exist
        if 'rating' not in ratings_df.columns:
            ratings_df['rating'] = 3.0  # Default rating
        self.ratingsDF = ratings_df

    def binarizeRatings(self, threshold=3.5):
        """Convert ratings to binary likes/dislikes"""
        if 'rating' not in self.ratingsDF.columns:
            raise ValueError("No rating column found")
        return (self.ratingsDF['rating'] >= threshold).astype(int)