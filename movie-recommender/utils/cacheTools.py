# add_to_cache.py

import os
import pandas as pd
import requests

OMDB_API_KEY = "6d810392"  
CACHE_FILE = "ml-100k/omdb_metadata.csv"  

def loadCachedData() -> pd.DataFrame:
    if os.path.exists(CACHE_FILE):
        print("✅ Loaded cached OMDb metadata.")
        return pd.read_csv(CACHE_FILE)
    else:
        print("📡 No cache found, starting fresh...")
        return pd.DataFrame()


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
    
def addMovieToCache(movie_data: dict) -> None:
    if movie_data is None:
        return

    # Load existing cache or initialize it
    df = loadCachedData()
    # Append new movie data to cache if it's not already present
    if movie_data["movieId"] not in df["movieId"].values:
        new_row = {
            "movieId": movie_data.get("movieId"),
            "title": movie_data.get("Title"),
            "genres": movie_data.get("Genre"),
            "directors": movie_data.get("Director"),
            "actors": movie_data.get("Actors"),
            "overview": movie_data.get("Plot"),
            "voteAverage": float(movie_data.get("imdbRating", 0))
        }
        # Append the new data to the dataframe and save it
        df = df.append(new_row, ignore_index=True)
        df.to_csv(CACHE_FILE, index=False)
        print(f"✅ Added movie {new_row['title']} to cache.")
    else:
        print(f"📦 Movie already in cache: {movie_data['movieId']}")

def getMovieData(imdb_id: str, movieId: int) -> dict:
    # Load cached data
    cached_data = loadCachedData()
    
    # Check if the movieId already exists in cache
    existing_movie = cached_data[cached_data["movieId"] == movieId]
    if not existing_movie.empty:
        return existing_movie.iloc[0].to_dict()
    
    # Fetch from OMDb if not found in cache
    movie_data = fetchMovieData(imdb_id)
    
    # If valid data, add to cache
    if movie_data:
        movie_data["movieId"] = movieId
        addMovieToCache(movie_data)
    
    return movie_data

def addMoviesFromLinks(linksPath: str) -> None:
    # Load movie links
    links = pd.read_csv(linksPath)

    for _, row in links.iterrows():
        imdb_id = row["imdbId"]
        movieId = row["movieId"]
        # Fetch movie data and add to cache
        getMovieData(imdb_id, movieId)

if __name__ == "__main__":
    addMoviesFromLinks("ml-100k/links.csv")
