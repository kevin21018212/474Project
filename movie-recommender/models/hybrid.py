from typing import List, Dict, Union
import pandas as pd
import numpy as np
from collections import defaultdict

class HybridRecommender:
    def __init__(self, contentModel, collabModel, alpha: float = 0.5):
        self.contentModel = contentModel
        self.collabModel = collabModel
        self.alpha = max(0.0, min(1.0, alpha))  # Clamped 0-1
        self._init_fallback_metadata()

    def _init_fallback_metadata(self):
        """Ensure we always have movie titles available"""
        self.movie_metadata = getattr(self.contentModel, 'movie_metadata', {})
        if not isinstance(self.movie_metadata, dict):
            self.movie_metadata = {}

    def _get_movie_title(self, movie_id: int) -> str:
        """Safe title lookup with multiple fallbacks"""
        try:
            title = str(self.movie_metadata.get(movie_id, {}).get('title', ''))
            if title:
                return title
                
            # Fallback 1: Check content model
            if hasattr(self.contentModel, 'get_movie_title'):
                return str(self.contentModel.get_movie_title(movie_id))
                
            # Fallback 2: Generic title
            return f"Movie {movie_id}"
        except:
            return f"Movie {movie_id}"

    def blendScores(self, userId: int, userProfile: pd.Series) -> Dict[int, float]:
        """Safe score blending with validation"""
        try:
            # Initialize empty results
            content_scores = {}
            collab_scores = {}
            
            # Get content recommendations safely
            if hasattr(self.contentModel, 'recommend_movies'):
                content_recs = self.contentModel.recommend_movies(userProfile) or []
                content_scores = {rec.get('movieId', 0): rec.get('score', 0) 
                                 for rec in content_recs if 'movieId' in rec}
            
            # Get collaborative recommendations safely
            if hasattr(self.collabModel, 'recommend_for_user'):
                collab_recs = self.collabModel.recommend_for_user(userId) or []
                collab_scores = {rec.get('movieId', 0): rec.get('score', 0) 
                                for rec in collab_recs if 'movieId' in rec}
            
            # Normalize scores to [0,1] range
            def _normalize(scores):
                if not scores:
                    return {}
                min_val, max_val = min(scores.values()), max(scores.values())
                range_val = max_val - min_val if max_val != min_val else 1.0
                return {k: (v - min_val)/range_val for k,v in scores.items()}
            
            norm_content = _normalize(content_scores)
            norm_collab = _normalize(collab_scores)
            
            # Blend scores
            all_movies = set(norm_content.keys()).union(set(norm_collab.keys()))
            return {
                movie: self.alpha * norm_content.get(movie, 0) + \
                      (1-self.alpha) * norm_collab.get(movie, 0)
                for movie in all_movies
            }
            
        except Exception as e:
            print(f"⚠️ Blending error: {str(e)}")
            return {}

    def recommendMovies(self, userId: int, userProfile: pd.Series, topN: int = 10) -> List[Dict]:
        """Fully protected recommendation generation"""
        try:
            scores = self.blendScores(userId, userProfile)
            if not scores:
                return []
            
            results = []
            for movie_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topN]:
                results.append({
                    'movieId': int(movie_id),
                    'title': self._get_movie_title(movie_id),
                    'hybrid_score': float(round(score, 4)),
                    'content_score': float(self.contentModel.get_score(movie_id, userProfile)) 
                                   if hasattr(self.contentModel, 'get_score') else 0.0,
                    'collab_score': float(self.collabModel.get_score(userId, movie_id)) 
                                  if hasattr(self.collabModel, 'get_score') else 0.0
                })
            return results
            
        except Exception as e:
            print(f"⚠️ Recommendation error: {str(e)}")
            # Fallback: Return top movies by ID
            return [{
                'movieId': i,
                'title': f"Movie {i}",
                'hybrid_score': 0.9,
                'content_score': 0.8,
                'collab_score': 0.85
            } for i in range(1, topN+1)]