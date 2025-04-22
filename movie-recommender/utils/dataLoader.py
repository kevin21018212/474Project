import pandas as pd
import requests
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

class IMDbLoader:
    def __init__(self, metadataPath: str):
        self.metadataPath = metadataPath
        self.metadataDF = None
        self.api_key = "6d810392"

    # Fetch metadata from OMDb API for a list of IMDb IDs
    def loadMetadata(self) -> pd.DataFrame:
        ids = pd.read_csv(self.metadataPath)["movie_id"]
        records = []
        for mid in ids:
            r = requests.get("https://www.omdbapi.com/", params={"apikey": self.api_key, "i": mid})
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
        self.metadataDF = pd.DataFrame(records)
        return self.metadataDF

    # Drop missing titles or plots and fill NaN ratings with 0
    def preprocessMetadata(self) -> pd.DataFrame:
        df = self.metadataDF
        return df.dropna(subset=["title", "overview"]).fillna({"vote_average": 0})


class MovieLensLoader:
    def __init__(self, ratingsPath: str):
        self.ratingsPath = ratingsPath
        self.ratingsDF = None

    # Load user–movie ratings from MovieLens
    def loadRatings(self) -> pd.DataFrame:
        self.ratingsDF = pd.read_csv(self.ratingsPath, sep="\t",
                                     names=["user_id", "movie_id", "rating", "timestamp"])
        return self.ratingsDF

    # Drop missing values and normalize ratings between 0 and 1 r
    def preprocessRatings(self) -> pd.DataFrame:
        df = self.ratingsDF.dropna()
        df["rating"] = MinMaxScaler().fit_transform(df[["rating"]])
        return df


class MetadataPreprocessor:
    def __init__(self, metadataDF: pd.DataFrame):
        self.metadataDF = metadataDF

    # Encode genres, directors, and actors 
    def encodeCategoricalFeatures(self) -> pd.DataFrame:
        mlb = MultiLabelBinarizer()
        genre = mlb.fit_transform(self.metadataDF["genres"])
        dirct = mlb.fit_transform(self.metadataDF["directors"])
        actrs = mlb.fit_transform(self.metadataDF["actors"])
        return pd.DataFrame(genre).join(pd.DataFrame(dirct)).join(pd.DataFrame(actrs))

    # Convert movie overviews into TF-IDF vectors 
    def applyTfidfToPlots(self) -> pd.DataFrame:
        vec = TfidfVectorizer(max_features=100, stop_words="english")
        mat = vec.fit_transform(self.metadataDF["overview"].fillna(""))
        return pd.DataFrame(mat.toarray(), columns=vec.get_feature_names_out())

    # Normalize vote_average field to 0–1 range 
    def normalizeNumericalFeatures(self) -> pd.DataFrame:
        scaled = MinMaxScaler().fit_transform(self.metadataDF[["vote_average"]])
        return pd.DataFrame(scaled, columns=["vote_avg_scaled"])


class RatingsPreprocessor:
    def __init__(self, ratingsDF: pd.DataFrame):
        self.ratingsDF = ratingsDF

    # Normalize rating values to a 0–1 scale
    def normalizeRatings(self) -> pd.DataFrame:
        self.ratingsDF["rating"] = MinMaxScaler().fit_transform(self.ratingsDF[["rating"]])
        return self.ratingsDF

    # Convert ratings into binary labels: 1 if >= threshold, 0 otherwise
    def binarizeRatings(self, threshold: float = 3.0) -> pd.DataFrame:
        self.ratingsDF["binary_rating"] = (self.ratingsDF["rating"] >= threshold).astype(int)
        return self.ratingsDF

    # Fill missing values in the ratings dataset using the most frequent value per column
    def fillMissingValues(self) -> pd.DataFrame:
        imputer = SimpleImputer(strategy="most_frequent")
        self.ratingsDF[:] = imputer.fit_transform(self.ratingsDF)
        return self.ratingsDF

