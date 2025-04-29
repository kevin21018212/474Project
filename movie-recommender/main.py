import pandas as pd
import numpy as np  
from utils.dataLoader import IMDbLoader, MovieLensLoader
from utils.dataLoader import MetadataPreprocessor, RatingsPreprocessor
from typing import List, Dict, Union
from models.contentFilter import ContentBasedRecommender
from models.collabFilter import CollaborativeRecommender
from models.hybrid import HybridRecommender
from utils.userProfile import UserProfile
from utils.dataLoader import IMDbLoader, MovieLensLoader, MetadataPreprocessor, RatingsPreprocessor


class DummyContentModel:
    def build_user_profile(self, *args):
        print("Warning: Using dummy content model")
        return pd.Series(dtype=float)
    
    def recommend_movies(self, *args):
        print("Warning: Using dummy content recommendations")
        return []

class DummyCollabModel:
    def recommend_for_user(self, *args):
        print("Warning: Using dummy collaborative recommendations")
        return []

class DummyHybridModel:
    def recommend(self, *args):
        print("Warning: Using dummy hybrid recommendations")
        return []
# Load metadata and ratings
def loadData():
    try:
        imdbLoader = IMDbLoader("ml-100k/links.csv", apiKey="your_api_key")
        metadataDf = imdbLoader.loadMetadata()
        
        # Ensure required columns exist
        if 'movieId' not in metadataDf.columns:
            if 'id' in metadataDf.columns:
                metadataDf = metadataDf.rename(columns={'id': 'movieId'})
            else:
                metadataDf['movieId'] = range(1, len(metadataDf)+1)
        
        metadataDf = imdbLoader.preprocessMetadata()
        
        # Load ratings
        movielensLoader = MovieLensLoader("ml-100k/ratings.csv")
        ratingsDf = movielensLoader.loadRatings()
        
        return metadataDf, ratingsDf
        
    except Exception as e:
        print(f"Data loading error: {e}")
        # Return empty DataFrames with required columns
        return pd.DataFrame(columns=['movieId', 'title']), \
               pd.DataFrame(columns=['userId', 'movieId', 'rating'])

def preprocessData(metadataDf: pd.DataFrame, ratingsDf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Safely preprocess data with comprehensive error handling
    Returns:
        tuple: (contentFeatures, binaryRatings) - guaranteed to return DataFrames
    """
    # Initialize default returns
    contentFeatures = pd.DataFrame()
    binaryRatings = pd.DataFrame()
    
    try:
        # ===== 1. Process Metadata =====
        metadataProcessor = MetadataPreprocessor(metadataDf)
        
        # Safely encode genres (with fallback)
        if 'genres' in metadataDf.columns:
            genres = metadataProcessor.encodeCategoricalFeatures()
        else:
            print("Warning: No genres column found")
            genres = pd.DataFrame(index=metadataDf.index)
        
        # Safely process text features
        text_features = pd.DataFrame()
        if any(col in metadataDf.columns for col in ['overview', 'title', 'description']):
            text_features = metadataProcessor.applyTfidfToPlots()
        
        # Safely process vote features
        vote_features = pd.DataFrame()
        if 'voteAverage' in metadataDf.columns:
            vote_features = metadataProcessor.normalizeVoteAverage()
        
        # Combine all available features
        contentFeatures = pd.concat(
            [genres, text_features, vote_features],
            axis=1
        ).fillna(0)

        # ===== 2. Process Ratings =====
        ratingsProcessor = RatingsPreprocessor(ratingsDf)
        
        # Ensure ratings column exists
        if 'rating' not in ratingsDf.columns:
            if 'binaryRating' in ratingsDf.columns:
                binaryRatings = ratingsDf['binaryRating']
            else:
                print("Warning: No rating column found, using default ratings")
                ratingsDf['rating'] = 3.0  # Neutral default
        
        binaryRatings = ratingsProcessor.binarizeRatings(threshold=3.5)
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        # Return empty DataFrames with correct indices if possible
        contentFeatures = pd.DataFrame(index=metadataDf.index if not metadataDf.empty else None)
        binaryRatings = pd.DataFrame(index=ratingsDf.index if not ratingsDf.empty else None)
    
    return contentFeatures, binaryRatings

#  user profile with favorite movies
def initializeUser(userId: int, favoriteMovieIds: list) -> UserProfile:
    user = UserProfile(userId)
    user.addFavorites(favoriteMovieIds)
    return user

# Train content and collaborative models
def trainModels(metadataDF, ratingsDF):
    """Train models with guaranteed return values"""
    try:
        # Initialize empty models as fallback
        content_model = None
        collab_model = None
        hybrid_model = None
        
        # Attempt to train each model
        if metadataDF is not None and not metadataDF.empty:
            content_model = ContentBasedRecommender(metadataDF)
        
        if ratingsDF is not None and not ratingsDF.empty:
            collab_model = CollaborativeRecommender(ratingsDF)
        
        if content_model and collab_model:
            hybrid_model = HybridRecommender(content_model, collab_model)
        
        return content_model, collab_model, hybrid_model
        
    except Exception as e:
        print(f"Model training error: {e}")
        # Return dummy models that won't cause attribute errors
        return DummyContentModel(), DummyCollabModel(), DummyHybridModel()

# Then to get recommendations:
metadataDf, ratingsDf = loadData()
contentFeatures, binaryRatings = preprocessData(metadataDf, ratingsDf)
content_model, collab_model, hybrid_model = trainModels(contentFeatures, binaryRatings)

favorite_movie_ids = []  # Replace with actual favorite movie IDs
user_profile = content_model.build_user_profile(favorite_movie_ids)
recommendations = content_model.recommend_movies(user_profile)
# Run hybrid recommendation pipeline
def runRecommendationPipeline(user: UserProfile, 
                           contentModel: ContentBasedRecommender,
                           collabModel: CollaborativeRecommender,
                           hybridModel: HybridRecommender) -> List[Dict[str, Union[int, float, str]]]:
    """
    Generate recommendations using all three approaches and combine results
    
    Args:
        user: User profile with preferences
        contentModel: Initialized content-based recommender
        collabModel: Initialized collaborative filtering recommender
        hybridModel: Initialized hybrid recommender
        
    Returns:
        List of recommended movies with scores from all approaches:
        [{
            'movieId': int,
            'title': str,
            'content_score': float,
            'collab_score': float,
            'hybrid_score': float
        }, ...]
    """
    # Get content-based recommendations
    user_profile = contentModel.build_user_profile(user.get_favorite_movies())
    content_recs = contentModel.recommend_movies(user_profile)
    
    # Get collaborative recommendations
    collab_recs = collabModel.recommend_for_user(user.user_id)
    
    # Get hybrid recommendations
    hybrid_recs = hybridModel.recommend(user.user_id)
    
    # Combine results into unified format
    all_recs = []
    recs_dict = {}
    
    # Process content recommendations
    for rec in content_recs:
        recs_dict[rec['movieId']] = {
            'movieId': rec['movieId'],
            'title': rec['title'],
            'content_score': rec['similarity'],
            'collab_score': 0,
            'hybrid_score': 0
        }
    
    # Add collaborative scores
    for rec in collab_recs:
        if rec['movieId'] in recs_dict:
            recs_dict[rec['movieId']]['collab_score'] = rec['score']
        else:
            recs_dict[rec['movieId']] = {
                'movieId': rec['movieId'],
                'title': rec['title'],
                'content_score': 0,
                'collab_score': rec['score'],
                'hybrid_score': 0
            }
    
    # Add hybrid scores
    for rec in hybrid_recs:
        if rec['movieId'] in recs_dict:
            recs_dict[rec['movieId']]['hybrid_score'] = rec['score']
        else:
            recs_dict[rec['movieId']] = {
                'movieId': rec['movieId'],
                'title': rec['title'],
                'content_score': 0,
                'collab_score': 0,
                'hybrid_score': rec['score']
            }
    
    return sorted(recs_dict.values(), 
                 key=lambda x: x['hybrid_score'], 
                 reverse=True)[:20]  # Return top 20


def evaluateResults(recommendations: List[Dict], groundTruth: List[int]) -> Dict[str, float]:
    """Safe evaluation with numpy import"""
    import numpy as np  # Ensure available locally
    
    if not recommendations or not groundTruth:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'ndcg': 0.0
        }
    
    recommended_ids = {rec['movieId'] for rec in recommendations}
    relevant_ids = set(groundTruth)
    
    # Basic metrics
    true_pos = len(recommended_ids & relevant_ids)
    false_pos = len(recommended_ids - relevant_ids)
    false_neg = len(relevant_ids - recommended_ids)
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # NDCG calculation with safety checks
    def dcg(scores):
        return sum((2**score - 1) / np.log2(i + 2) for i, score in enumerate(scores))
    
    try:
        ideal_scores = [1] * min(len(relevant_ids), len(recommendations))
        actual_scores = [1 if rec['movieId'] in relevant_ids else 0 
                        for rec in recommendations]
        ndcg = dcg(actual_scores) / dcg(ideal_scores) if dcg(ideal_scores) > 0 else 0
    except:
        ndcg = 0.0
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'ndcg': round(ndcg, 4)
    }


def main():
    # 1. Data Loading (keep existing)
    metadataDf, ratingsDf = loadData()
    contentFeatures, binaryRatings = preprocessData(metadataDf, ratingsDf)
    
    # 2. Model Training (keep existing)
    content_model, collab_model, hybrid_model = trainModels(metadataDf, binaryRatings)
    
    # 3. NEW: Recommendation Generation
    if None in (content_model, collab_model, hybrid_model):
        print("Error: Some models failed to train")
        return
    
    try:
        # User setup
        test_user_id = 1
        favorite_movie_ids = get_favorites(binaryRatings, test_user_id)
        
        # Content-based recommendations
        if hasattr(content_model, 'build_user_profile'):
            user_profile = content_model.build_user_profile(favorite_movie_ids)
            recommendations = content_model.recommend_movies(user_profile)
        else:
            recommendations = []
            
        # Display results
        print_recommendations(recommendations)
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")
    
    # 4. Evaluation (keep existing)
    if recommendations:
        metrics = evaluateResults(recommendations, favorite_movie_ids)
        print_metrics(metrics)

# NEW HELPER FUNCTIONS
def get_favorites(ratings_df, user_id, min_favorites=3):
    """Safely get favorite movies with fallback"""
    try:
        favorites = ratings_df[ratings_df[user_id] == 1].index.tolist()
        return favorites if favorites else ratings_df.index[:min_favorites]
    except:
        return ratings_df.index[:min_favorites]

def print_recommendations(recommendations, top_n=10):
    """Safe recommendation printing"""
    print("\nTop Recommendations:")
    for i, rec in enumerate(recommendations[:top_n], 1):
        print(f"{i}. {rec.get('title', 'Unknown')} (Score: {rec.get('score', 0):.2f})")

def print_metrics(metrics):
    """Safe metric printing"""
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()