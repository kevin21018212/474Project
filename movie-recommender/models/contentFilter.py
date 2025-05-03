import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Union
import re

class ContentBasedRecommender:
    def __init__(self, metadata_df: pd.DataFrame):
        self.metadata = metadata_df.set_index('movieId')
        self.feature_matrix = self._build_feature_matrix()
    
    def _build_feature_matrix(self) -> pd.DataFrame:
        """Build features with guaranteed non-empty output"""
        # Ensure basic features exist
        features = []
        
        # 1. Genre features (if available)
        if 'genres' in self.metadata.columns:
            genres = self.metadata['genres'].str.get_dummies(sep='|')
            features.append(genres)
        
        # 2. Basic text features (fallback to title if no overview)
        text_col = 'overview' if 'overview' in self.metadata.columns else 'title'
        if text_col in self.metadata.columns:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(
                stop_words='english',
                min_df=2,
                token_pattern=r'\b\w{3,}\b'  # Only words with 3+ chars
            )
            try:
                text_features = tfidf.fit_transform(self.metadata[text_col].fillna(''))
                text_df = pd.DataFrame(
                    text_features.toarray(),
                    index=self.metadata.index,
                    columns=[f"tfidf_{i}" for i in range(text_features.shape[1])]
                )
                features.append(text_df)
            except ValueError:
                pass
        
        # Combine all available features
        if features:
            return pd.concat(features, axis=1).fillna(0)
        else:
            # Fallback: use movie IDs as minimal features
            return pd.DataFrame(
                np.eye(len(self.metadata)),
                index=self.metadata.index
            )
    
    def build_user_profile(self, favorite_movie_ids: List[int]) -> pd.Series:
        """Create normalized user profile vector"""
        valid_ids = [mid for mid in favorite_movie_ids if mid in self.feature_matrix.index]
        
        if not valid_ids:
            return pd.Series(0, index=self.feature_matrix.columns)
        
        profile = self.feature_matrix.loc[valid_ids].mean(axis=0)
        # Normalize profile
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile
    
    def recommend_movies(self, user_profile: pd.Series, top_n: int = 10) -> List[Dict]:
        """Get recommendations with guaranteed non-empty results"""
        try:
            # Ensure valid input shapes
            if len(user_profile) == 0 or self.feature_matrix.shape[1] == 0:
                return self._fallback_recommendations(top_n)
                
            similarities = cosine_similarity(
                user_profile.values.reshape(1, -1),
                self.feature_matrix
            )[0]
            
            # Get top recommendations
            movie_ids = self.feature_matrix.index
            top_indices = np.argsort(similarities)[-top_n:][::-1]
            
            return [{
                'movieId': movie_ids[i],
                'title': self.metadata.loc[movie_ids[i], 'title'],
                'score': float(similarities[i])
            } for i in top_indices]
            
        except Exception:
            return self._fallback_recommendations(top_n)
    
    def _fallback_recommendations(self, top_n: int) -> List[Dict]:
        """Fallback when proper recommendations fail"""
        return [{
            'movieId': mid,
            'title': self.metadata.loc[mid, 'title'],
            'score': 1.0
        } for mid in self.metadata.index[:top_n]]

    
    def update_user_profile(self, user_profile: pd.Series, movie_id: int, 
                          liked: bool = True, weight: float = 0.2) -> pd.Series:
        """
        Update user profile with new feedback
        
        Args:
            user_profile: Current user profile vector
            movie_id: Movie to incorporate feedback for
            liked: Whether the user liked the movie
            weight: How much to weight the new feedback (0-1)
            
        Returns:
            Updated user profile vector
        """
        movie_features = self.feature_matrix.loc[movie_id]
        if liked:
            return (1 - weight) * user_profile + weight * movie_features
        else:
            return (1 + weight) * user_profile - weight * movie_features
    @property
    def movie_metadata(self):
        """Safe access to metadata with fallback"""
        if not hasattr(self, '_movie_metadata'):
            self._movie_metadata = {}
        return self._movie_metadata
    
    def get_movie_title(self, movie_id: int) -> str:
        """Safe title lookup"""
        return str(self.movie_metadata.get(movie_id, {}).get('title', f"Movie {movie_id}"))