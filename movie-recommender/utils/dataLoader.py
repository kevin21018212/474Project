import pandas as pd
import requests
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import time


class IMDbLoader:
    def __init__(self, linksPath: str):
        self.linksPath = linksPath
        self.metadataDF = None
        self.api_key = "6d810392"

    # Fetch metadata from OMDb using IMDb IDs
    def loadMetadata(self) -> pd.DataFrame:
    # Check for cached metadata
        try:
            self.metadataDF = pd.read_csv("ml-100k/omdb_metadata.csv")
            print("✅ Loaded cached OMDb metadata.")
            return self.metadataDF
        except FileNotFoundError:
            print("📡 No cache found. Fetching from OMDb API...")

        # Fetch from API
        links = pd.read_csv(self.linksPath).head(100)
        links["imdb_id"] = links["imdbId"].apply(lambda x: f"tt{int(x):07d}")

        records = []
        for _, row in links.iterrows():
            mid = row["movieId"]
            imdb = row["imdb_id"]
            try:
                r = requests.get("https://www.omdbapi.com/", params={"apikey": self.api_key, "i": imdb}, timeout=5)
                if r.status_code != 200 or r.json().get("Response") == "False":
                    continue
                d = r.json()
                records.append({
                    "movie_id": mid,
                    "title": d.get("Title", ""),
                    "genres": d.get("Genre", "").split(", "),
                    "directors": d.get("Director", "").split(", "),
                    "actors": d.get("Actors", "").split(", ")[:3],
                    "overview": d.get("Plot", ""),
                    "vote_average": float(d.get("imdbRating", 0)) if d.get("imdbRating", "N/A") != "N/A" else 0
                })
            except Exception as e:
                print(f" Error fetching {imdb}: {e}")

        self.metadataDF = pd.DataFrame(records)

        # Cache for next time
        self.metadataDF.to_csv("ml-100k/omdb_metadata.csv", index=False)
        return self.metadataDF


    # Clean metadata (drop missing titles/plots)
    def preprocessMetadata(self) -> pd.DataFrame:
        df = self.metadataDF
        return df.dropna(subset=["title", "overview"]).fillna({"vote_average": 0})


class MovieLensLoader:
    def __init__(self, ratingsPath: str):
        self.ratingsPath = ratingsPath
        self.ratingsDF = None

    # Load MovieLens ratings.csv (standard CSV)
    def loadRatings(self) -> pd.DataFrame:
        self.ratingsDF = pd.read_csv(self.ratingsPath)  # No sep="\t"
        return self.ratingsDF

    # Drop missing and normalize ratings
    def preprocessRatings(self) -> pd.DataFrame:
        df = self.ratingsDF.dropna()
        if df[["rating"]].empty:
            raise ValueError("No valid ratings found to normalize.")
        df["rating"] = MinMaxScaler().fit_transform(df[["rating"]])
        return df


class MetadataPreprocessor:
    def __init__(self, metadataDF: pd.DataFrame):
        self.metadataDF = metadataDF

    # Encode genres, directors, and actors
    def encodeCategoricalFeatures(self) -> pd.DataFrame:
        mlb = MultiLabelBinarizer()

        genre = mlb.fit_transform(self.metadataDF["genres"])
        genre_df = pd.DataFrame(genre, columns=[f"genre_{g}" for g in mlb.classes_])

        dirct = mlb.fit_transform(self.metadataDF["directors"])
        dirct_df = pd.DataFrame(dirct, columns=[f"director_{d}" for d in mlb.classes_])

        actrs = mlb.fit_transform(self.metadataDF["actors"])
        actrs_df = pd.DataFrame(actrs, columns=[f"actor_{a}" for a in mlb.classes_])

        return genre_df.join(dirct_df).join(actrs_df)

    # TF-IDF vectorize plots
    def applyTfidfToPlots(self) -> pd.DataFrame:
        vec = TfidfVectorizer(max_features=100, stop_words="english")
        mat = vec.fit_transform(self.metadataDF["overview"].fillna(""))
        return pd.DataFrame(mat.toarray(), columns=vec.get_feature_names_out())

    # Normalize vote_average
    def normalizeNumericalFeatures(self) -> pd.DataFrame:
        scaled = MinMaxScaler().fit_transform(self.metadataDF[["vote_average"]])
        return pd.DataFrame(scaled, columns=["vote_avg_scaled"])


class RatingsPreprocessor:
    def __init__(self, ratingsDF: pd.DataFrame):
        self.ratingsDF = ratingsDF

    # Normalize rating values
    def normalizeRatings(self) -> pd.DataFrame:
        self.ratingsDF["rating"] = MinMaxScaler().fit_transform(self.ratingsDF[["rating"]])
        return self.ratingsDF

    # Binarize ratings based on threshold
    def binarizeRatings(self, threshold: float = 3.0) -> pd.DataFrame:
        self.ratingsDF["binary_rating"] = (self.ratingsDF["rating"] >= threshold).astype(int)
        return self.ratingsDF

    # Fill missing values
    def fillMissingValues(self) -> pd.DataFrame:
        imputer = SimpleImputer(strategy="most_frequent")
        self.ratingsDF[:] = imputer.fit_transform(self.ratingsDF)
        return self.ratingsDF
