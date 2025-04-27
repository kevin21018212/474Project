import pandas as pd
import requests
import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.helpers import normalizeVectors

# Fetch and preprocess movie metadata 
class IMDbLoader:
    def __init__(self, linksPath: str):
        self.linksPath = linksPath
        self.metadataDF = None
        self.apiKey = "6d810392"

    # Fetch metadata from OMDb using IMDb IDs
    def loadMetadata(self) -> pd.DataFrame:
    # Check for cached metadata
        try:
            self.metadataDF = pd.read_csv("ml-100k/omdb_metadata.csv")
            print(" Loaded cached OMDb metadata.")
        except FileNotFoundError:
            print(" No cache found. Fetching from OMDb API...")

            links = pd.read_csv(self.linksPath).head(400)
            links["imdbIdFormatted"] = links["imdbId"].apply(lambda x: f"tt{int(x):07d}")

            records = []
            for _, row in links.iterrows():
                imdbId = row["imdbIdFormatted"]
                try:
                    response = requests.get(
                        "https://www.omdbapi.com/",
                        params={"apikey": self.apiKey, "i": imdbId},
                        timeout=5
                    )
                    data = response.json()
                    if data.get("Response") == "False":
                        continue
                    records.append({
                        "movieId": row["movieId"],
                        "title": data.get("Title", ""),
                        "genres": data.get("Genre", "").split(", "),
                        "directors": data.get("Director", "").split(", "),
                        "actors": data.get("Actors", "").split(", ")[:3],
                        "overview": data.get("Plot", ""),
                        "voteAverage": float(data.get("imdbRating", 0)) if data.get("imdbRating") != "N/A" else 0
                    })
                except Exception as e:
                    print(f"Error fetching {imdbId}: {e}")

            self.metadataDF = pd.DataFrame(records)
            self.metadataDF.to_csv("ml-100k/omdb_metadata.csv", index=False)

        #Rename to camelcase
        self.metadataDF.rename(columns={
            "movie_id": "movieId",
            "vote_average": "voteAverage"
        }, inplace=True)

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
