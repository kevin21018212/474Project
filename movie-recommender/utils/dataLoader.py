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


class MetadataPreprocessor:
    def __init__(self, metadata_df: pd.DataFrame):
        """
        Initialize with movie metadata DataFrame
        Required columns: 'movieId', at least one of ['genres', 'title', 'overview', 'voteAverage']
        """
        self.df = metadata_df.copy()
        
        # Ensure movieId exists
        if 'movieId' not in self.df.columns:
            if 'id' in self.df.columns:
                self.df.rename(columns={'id': 'movieId'}, inplace=True)
            else:
                self.df['movieId'] = range(1, len(self.df)+1)

    def encodeCategoricalFeatures(self) -> pd.DataFrame:
        """Convert genres from strings to one-hot encoded features"""
        try:
            if 'genres' not in self.df.columns:
                return pd.DataFrame(index=self.df.index)
                
            # Handle pipe-separated genres (e.g., "Action|Adventure")
            split_genres = self.df['genres'].str.split('|')
            
            mlb = MultiLabelBinarizer()
            genre_features = pd.DataFrame(
                mlb.fit_transform(split_genres),
                columns=[f"genre_{g}" for g in mlb.classes_],
                index=self.df.index
            )
            return genre_features
            
        except Exception as e:
            print(f"⚠️ Genre encoding failed: {e}")
            return pd.DataFrame(index=self.df.index)

    def applyTfidfToPlots(self, text_columns: list = None) -> pd.DataFrame:
        """Convert text features to TF-IDF vectors"""
        try:
            text_columns = text_columns or ['overview', 'title']
            available_cols = [col for col in text_columns if col in self.df.columns]
            
            if not available_cols:
                return pd.DataFrame(index=self.df.index)
                
            # Combine text features
            combined_text = self.df[available_cols[0]].fillna('')
            for col in available_cols[1:]:
                combined_text += ' ' + self.df[col].fillna('')
                
            tfidf = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_features = tfidf.fit_transform(combined_text)
            
            return pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f"tfidf_{i}" for i in range(tfidf_features.shape[1])],
                index=self.df.index
            )
        except Exception as e:
            print(f"⚠️ TF-IDF processing failed: {e}")
            return pd.DataFrame(index=self.df.index)

    def normalizeVoteAverage(self) -> pd.DataFrame:
        """Normalize vote averages to 0-1 scale"""
        try:
            if 'voteAverage' not in self.df.columns:
                return pd.DataFrame(index=self.df.index)
                
            votes = self.df['voteAverage'].fillna(0)
            normalized = (votes - votes.min()) / (votes.max() - votes.min())
            return pd.DataFrame({'norm_votes': normalized}, index=self.df.index)
            
        except Exception as e:
            print(f"⚠️ Vote normalization failed: {e}")
            return pd.DataFrame(index=self.df.index)

    def process_all_features(self) -> pd.DataFrame:
        """
        Master method that combines all feature types.
        Returns:
            DataFrame with all processed features, indexed by movieId
        """
        try:
            features = []
            
            # 1. Genre Features
            genre_features = self.encodeCategoricalFeatures()
            if not genre_features.empty:
                features.append(genre_features)
            
            # 2. Text Features
            text_features = self.applyTfidfToPlots()
            if not text_features.empty:
                features.append(text_features)
            
            # 3. Numerical Features
            vote_features = self.normalizeVoteAverage()
            if not vote_features.empty:
                features.append(vote_features)
            
            # Combine all features
            if features:
                combined = pd.concat(features, axis=1)
                combined['movieId'] = self.df['movieId']  # Ensure movieId is preserved
                return combined.fillna(0)
            else:
                print("⚠️ No features generated - returning minimal DataFrame")
                return pd.DataFrame({'movieId': self.df['movieId']})
                
        except Exception as e:
            print(f"❌ Critical error in feature processing: {e}")
            return pd.DataFrame({'movieId': self.df['movieId']})

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