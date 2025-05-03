import pandas as pd
from typing import List, Dict, Union
from models.contentFilter import ContentBasedRecommender
from models.collabFilter import CollaborativeRecommender
from models.hybrid import HybridRecommender
from utils.userProfile import UserProfile
from utils.dataLoader import IMDbLoader, MovieLensLoader, MetadataPreprocessor, RatingsPreprocessor

class DummyContentModel:
    """Fallback model if content filtering fails"""
    def build_user_profile(self, *args):
        print("Warning: Using dummy content model")
        return pd.Series(dtype=float)
    
    def recommend_movies(self, *args):
        return []
    
    def get_score(self, *args):
        return 0.0

class DummyCollabModel:
    """Fallback model if collaborative filtering fails"""
    def recommend_for_user(self, *args):
        return []
    
    def get_score(self, *args):
        return 0.0

def loadData() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and validates movie metadata and ratings data.
    
    Returns:
        tuple: (metadata_df, ratings_df) - Guaranteed to return DataFrames 
              with required columns, even if empty.
    """
    # Initialize empty DataFrames with required columns
    empty_metadata = pd.DataFrame(columns=['movieId', 'title', 'genres'])
    empty_ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
    
    try:
        # --- 1. Load Metadata ---
        print("Loading metadata...")
        imdbLoader = IMDbLoader("ml-100k/links.csv", apiKey="your_api_key")
        metadata_df = imdbLoader.loadMetadata()
        
        # Validate minimum metadata columns
        if not {'movieId', 'title'}.issubset(metadata_df.columns):
            print("⚠️ Missing required columns in metadata - attempting fixes...")
            if 'id' in metadata_df.columns:
                metadata_df = metadata_df.rename(columns={'id': 'movieId'})
            else:
                print("⚠️ No movieId column - generating dummy IDs")
                metadata_df['movieId'] = range(1, len(metadata_df)+1)
        
        # Add default genres if missing
        if 'genres' not in metadata_df.columns:
            metadata_df['genres'] = 'Unknown'
        
        # --- 2. Load Ratings ---
        print("Loading ratings...")
        movielensLoader = MovieLensLoader("ml-100k/ratings.csv")
        ratings_df = movielensLoader.loadRatings()
        
        # Validate ratings columns
        required_rating_cols = {'userId', 'movieId', 'rating'}
        missing_cols = required_rating_cols - set(ratings_df.columns)
        if missing_cols:
            print(f"⚠️ Missing columns in ratings: {missing_cols}")
            if 'user_id' in ratings_df.columns:  # Common alternative
                ratings_df = ratings_df.rename(columns={'user_id': 'userId'})
            if 'movie_id' in ratings_df.columns:
                ratings_df = ratings_df.rename(columns={'movie_id': 'movieId'})
            
            # Final check if still missing
            missing_cols = required_rating_cols - set(ratings_df.columns)
            if missing_cols:
                raise KeyError(f"Could not resolve missing columns: {missing_cols}")
        
        # --- 3. Data Consistency Checks ---
        # Ensure no NaN in critical columns
        for col in ['movieId', 'userId', 'rating']:
            if col in ratings_df.columns and ratings_df[col].isna().any():
                print(f"⚠️ NaN values in {col} - filling with defaults")
                if col == 'rating':
                    ratings_df[col] = ratings_df[col].fillna(3.0)  # Neutral rating
                else:
                    ratings_df = ratings_df.dropna(subset=[col])
        
        # Ensure movieId consistency between datasets
        common_movies = set(metadata_df['movieId']).intersection(set(ratings_df['movieId']))
        if len(common_movies) == 0:
            print("⚠️ No common movies between metadata and ratings!")
        
        print("✅ Data loaded successfully")
        return metadata_df, ratings_df
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return empty_metadata, empty_ratings
    except Exception as e:
        print(f"❌ Unexpected error in loadData: {e}")
        return empty_metadata, empty_ratings

def preprocessData(metadataDf: pd.DataFrame, ratingsDf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare features and ratings"""
    try:
        # Process metadata
        processor = MetadataPreprocessor(metadataDf)
        contentFeatures = processor.process_all_features()  # Should handle genres/text
        
        # Process ratings
        ratingsProcessor = RatingsPreprocessor(ratingsDf)
        binaryRatings = ratingsProcessor.binarizeRatings(threshold=3.5)
        
        return contentFeatures, binaryRatings
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return pd.DataFrame(), pd.DataFrame()

def trainModels(contentFeatures, binaryRatings):
    """Train models with guaranteed returns"""
    try:
        # Initialize models
        content_model = ContentBasedRecommender(contentFeatures)
        collab_model = CollaborativeRecommender(binaryRatings)
        
        # Build metadata (safe for any DataFrame)
        try:
            if not contentFeatures.empty and 'movieId' in contentFeatures.columns:
                content_model.movie_metadata = contentFeatures.set_index('movieId') \
                    .apply(lambda x: x.to_dict(), axis=1) \
                    .to_dict()
        except Exception as e:
            print(f"⚠️ Metadata creation failed: {str(e)}")
        
        # Verify models are functional
        if (hasattr(content_model, 'recommend_movies') and 
            hasattr(collab_model, 'recommend_for_user')):
            return content_model, collab_model, HybridRecommender(content_model, collab_model)
            
        raise RuntimeError("Essential model methods missing")
        
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        return DummyContentModel(), DummyCollabModel(), None

def get_user_favorites(ratingsDf: pd.DataFrame, userId: int, min_favorites: int = 3) -> List[int]:
    """Get user's favorite movies with robust error handling"""
    try:
        # Convert to DataFrame if Series
        if isinstance(ratingsDf, pd.Series):
            ratingsDf = ratingsDf.to_frame().T
            
        # Check if DataFrame is valid
        if ratingsDf.empty or 'movieId' not in ratingsDf.columns:
            return [1, 2, 3][:min_favorites]  # Fallback IDs
            
        # Get user's high-rated movies
        user_ratings = ratingsDf[(ratingsDf['userId'] == userId) & 
                               (ratingsDf['rating'] >= 4)]
                               
        if len(user_ratings) >= min_favorites:
            return user_ratings['movieId'].tolist()[:min_favorites]
            
        # Fallback to popular movies
        popular = ratingsDf['movieId'].value_counts().index.tolist()
        return popular[:min_favorites]
        
    except Exception as e:
        print(f"⚠️ Error in get_user_favorites: {str(e)}")
        return [1, 2, 3][:min_favorites]  # Default movie IDs

def evaluate(recommendations: List[Dict], true_positives: List[int]) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    recommended_ids = {r['movieId'] for r in recommendations}
    relevant_ids = set(true_positives)
    
    tp = len(recommended_ids & relevant_ids)
    precision = tp / len(recommended_ids) if recommended_ids else 0
    recall = tp / len(relevant_ids) if relevant_ids else 0
    
    return {
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1': round(2*(precision*recall)/(precision+recall), 3) if (precision+recall) else 0
    }

def print_results(recommendations: List[Dict], metrics: Dict[str, float]):
    """Display recommendations and metrics"""
    print("\n=== Top Recommendations ===")
    for i, rec in enumerate(recommendations[:10], 1):
        print(f"{i}. {rec.get('title', 'Unknown')}")
        print(f"   Score: {rec.get('hybrid_score', 0):.2f} "
              f"(Content: {rec.get('content_score', 0):.2f}, "
              f"Collab: {rec.get('collab_score', 0):.2f})")
    
    print("\n=== Evaluation ===")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.3f}")

def main():
    try:
        print("🚀 Starting recommendation system...")
        
        # ======================
        # 1. Data Loading
        # ======================
        print("\n🔍 Loading data...")
        metadata_df, ratings_df = loadData()
        
        if ratings_df.empty or metadata_df.empty:
            print("⚠️ Empty data detected - using sample dataset")
            metadata_df = pd.DataFrame({
                'movieId': [1, 2, 3],
                'title': ['Movie A', 'Movie B', 'Movie C'],
                'genres': ['Action|Adventure', 'Comedy', 'Drama']
            })
            ratings_df = pd.DataFrame({
                'userId': [1, 1, 2, 2],
                'movieId': [1, 2, 1, 3],
                'rating': [5, 4, 5, 3]
            })

        # ======================
        # 2. Data Preprocessing
        # ======================
        print("\n🛠️ Preprocessing data...")
        try:
            # Process metadata
            metadata_processor = MetadataPreprocessor(metadata_df)
            content_features = metadata_processor.process_all_features()
            
            # Process ratings
            ratings_processor = RatingsPreprocessor(ratings_df)
            binary_ratings = ratings_processor.binarizeRatings(threshold=3.5)
        except Exception as e:
            print(f"❌ Preprocessing failed: {str(e)}")
            raise

        # ======================
        # 3. Model Training
        # ======================
        print("\n🧠 Training models...")
        content_model, collab_model, hybrid_model = trainModels(
            content_features,
            binary_ratings
        )
        
        if hybrid_model is None:
            print("⚠️ Using fallback recommendations")
            return show_recommendations(generate_fallback_recommendations())

        # ======================
        # 4. User Setup
        # ======================
        test_user_id = 1
        try:
            favorites = get_user_favorites(binary_ratings, test_user_id)
            user = UserProfile(userId=test_user_id)
            user.addFavorites(favorites if favorites else [1, 2, 3])
        except Exception as e:
            print(f"❌ User setup failed: {str(e)}")
            user = UserProfile(userId=test_user_id)
            user.addFavorites([1, 2, 3])

        # ======================
        # 5. Recommendations
        # ======================
        print("\n✨ Generating recommendations...")
        try:
            recommendations = hybrid_model.recommendMovies(
                userId=user.userId,
                userProfile=content_model.build_user_profile(user.get_favorite_movies()),
                topN=10
            )
        except Exception as e:
            print(f"⚠️ Recommendation failed: {str(e)}")
            recommendations = generate_fallback_recommendations()

        # ======================
        # 6. Display Results
        # ======================
        show_recommendations(recommendations)
        
    except Exception as e:
        print(f"\n💥 Critical system failure: {str(e)}")
        print("🛑 Shutting down...")
        return []

# Helper Functions
def show_recommendations(recommendations: List[Dict]):
    """Display recommendations with formatting"""
    if not recommendations:
        print("❌ No recommendations available")
        return
    
    print("\n🎬 Top Recommendations:")
    for i, rec in enumerate(recommendations[:10], 1):
        print(f"{i}. {rec.get('title', 'Unknown Movie')}")
        print(f"   Hybrid Score: {rec.get('hybrid_score', 0):.2f}")
        print(f"   Content Similarity: {rec.get('content_score', 0):.2f}")
        print(f"   User Match: {rec.get('collab_score', 0):.2f}")
        print()

def generate_fallback_recommendations(topN: int = 10) -> List[Dict]:
    """Generate recommendations when models fail"""
    return [{
        'movieId': i,
        'title': f"Popular Movie {i}",
        'hybrid_score': round(0.9 - (i*0.02), 2),
        'content_score': 0.8,
        'collab_score': 0.85
    } for i in range(1, topN+1)]

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()