# add_to_cache.py
import os
import pandas as pd
import requests

OMDB_API_KEY = "6d810392"  
CACHE_FILE = "ml-100k/omdb_metadata.csv"  

# Load previously cached metadata if it exists
def loadCachedData() -> pd.DataFrame:
    if os.path.exists(CACHE_FILE):
        print("✅ Loaded cached OMDb metadata.")
        return pd.read_csv(CACHE_FILE)
    else:
        print("📡 No cache found, starting fresh...")
        return pd.DataFrame()

# Fetch movie metadata from OMDb API using an IMDb ID
def fetchMovieData(imdb_id: str) -> dict:
    url = f"https://www.omdbapi.com/"
    params = {"apikey": OMDB_API_KEY, "i": imdb_id}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("Response") == "True":
            return data
        else:
            print(f"Error fetching data for {imdb_id}: {data.get('Error')}")
            return None
    except Exception as e:
        print(f"Error fetching data for {imdb_id}: {e}")
        return None

# Add new movie metadata to the local cache if not already present
def addMovieToCache(movie_data: dict) -> None:
    if movie_data is None:
        return

    df = loadCachedData()
    if movie_data["movieId"] not in df.get("movieId", pd.Series()).values:
        new_row = {
            "movieId": movie_data.get("movieId"),
            "title": movie_data.get("Title"),
            "genres": movie_data.get("Genre"),
            "directors": movie_data.get("Director"),
            "actors": movie_data.get("Actors"),
            "overview": movie_data.get("Plot"),
            "voteAverage": float(movie_data.get("imdbRating", 0))
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CACHE_FILE, index=False)
        print(f"✅ Added movie {new_row['title']} to cache.")
    else:
        print(f"📦 Movie already in cache: {movie_data['movieId']}")

# Fetch metadata (using cache if possible) for a movie by IMDb ID and MovieLens ID
def getMovieData(imdb_id: str, movieId: int) -> dict:
    cached_data = loadCachedData()
    existing_movie = cached_data[cached_data["movieId"] == movieId]
    if not existing_movie.empty:
        return existing_movie.iloc[0].to_dict()

    movie_data = fetchMovieData(imdb_id)
    if movie_data:
        movie_data["movieId"] = movieId
        addMovieToCache(movie_data)
    return movie_data

# Process all movies in the MovieLens links file and add them to the cache
def addMoviesFromLinks(linksPath: str) -> None:
    links = pd.read_csv(linksPath)
    for _, row in links.iterrows():
        imdb_id = row["imdbId"]
        movieId = row["movieId"]
        getMovieData(imdb_id, movieId)

if __name__ == "__main__":
    addMoviesFromLinks("ml-100k/links.csv")
