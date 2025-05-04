import requests
import pandas as pd
import os

class OmdbFetcher:
    def __init__(self, apiKey: str, cachePath: str = "ml-100k/omdb_metadata.csv"):
        self.apiKey = apiKey
        self.cachePath = cachePath
        self.cacheDF = self._loadCache()

    def _loadCache(self):
        if os.path.exists(self.cachePath):
            return pd.read_csv(self.cachePath)
        else:
            return pd.DataFrame(columns=["movieId", "title", "genres", "directors", "actors", "overview", "voteAverage"])

    def saveCache(self):
        self.cacheDF.to_csv(self.cachePath, index=False)

    def fetchMovie(self, movieId: int, imdbId: int) -> dict:
        imdbFormatted = f"tt{int(imdbId):07d}"

        # Check if movieId is in the cache
        if movieId in self.cacheDF["movieId"].values:
            cachedRow = self.cacheDF[self.cacheDF["movieId"] == movieId].iloc[0].to_dict()
            cachedRow["genres"] = eval(cachedRow["genres"])
            cachedRow["directors"] = eval(cachedRow["directors"])
            cachedRow["actors"] = eval(cachedRow["actors"])
            return cachedRow

        # Fetch from OMDb API if not in cache
        response = requests.get(
            "https://www.omdbapi.com/",
            params={"apikey": self.apiKey, "i": imdbFormatted},
            timeout=5
        )
    
        data = response.json()
        if data.get("Response") == "False":
            print(f"❌ Error fetching movie {movieId} from OMDb: {data.get('Error', 'Unknown error')}")
            return {}

        movieData = {
            "movieId": movieId,
            "title": data.get("Title", ""),
            "genres": data.get("Genre", "").split(", "),
            "directors": data.get("Director", "").split(", "),
            "actors": data.get("Actors", "").split(", ")[:3],
            "overview": data.get("Plot", ""),
            "voteAverage": float(data.get("imdbRating", 0)) if data.get("imdbRating") != "N/A" else 0
        }

        # Append to cache
        cacheEntry = movieData.copy()
        cacheEntry["genres"] = str(cacheEntry["genres"])
        cacheEntry["directors"] = str(cacheEntry["directors"])
        cacheEntry["actors"] = str(cacheEntry["actors"])
        self.cacheDF = pd.concat([self.cacheDF, pd.DataFrame([cacheEntry])], ignore_index=True)
        self.saveCache()

        return movieData

    def getMovieTitle(self, movieId: int) -> str:
        # Check if title is already cached
        if movieId in self.cacheDF["movieId"].values:
            return self.cacheDF.loc[self.cacheDF["movieId"] == movieId, "title"].values[0]

        # If not in cache, try fetching from OMDb
        try:
            linksDF = pd.read_csv("ml-100k/links.csv")
            imdbId = linksDF.loc[linksDF["movieId"] == movieId, "imdbId"].values
            if len(imdbId) == 0:
                return "Unknown Title"
            imdbId = imdbId[0]

            data = self.fetchMovie(movieId, imdbId)
            return data.get("title", "Unknown Title") if data else "Unknown Title"
        except Exception as e:
            print(f"❌ Failed to fetch title for movieId {movieId}: {e}")
            return "Unknown Title"

    def addMovieByTitle(self, title: str) -> dict:
        # Query OMDb API by title
        response = requests.get(
            "https://www.omdbapi.com/",
            params={"apikey": self.apiKey, "t": title},
            timeout=5
        )
        data = response.json()
        if data.get("Response") == "False":
            print(f"❌ OMDb could not find: {title}")
            return None

        # Generate a new movieId
        newMovieId = int(self.cacheDF["movieId"].max() + 1) if not self.cacheDF.empty else 100000
        movieData = {
            "movieId": newMovieId,
            "title": data.get("Title", title),
            "genres": data.get("Genre", "").split(", "),
            "directors": data.get("Director", "").split(", "),
            "actors": data.get("Actors", "").split(", ")[:3],
            "overview": data.get("Plot", ""),
            "voteAverage": float(data.get("imdbRating", 0)) if data.get("imdbRating") != "N/A" else 0
        }

        # Append to cache
        cacheEntry = movieData.copy()
        cacheEntry["genres"] = str(cacheEntry["genres"])
        cacheEntry["directors"] = str(cacheEntry["directors"])
        cacheEntry["actors"] = str(cacheEntry["actors"])
        self.cacheDF = pd.concat([self.cacheDF, pd.DataFrame([cacheEntry])], ignore_index=True)
        self.saveCache()

        return movieData
