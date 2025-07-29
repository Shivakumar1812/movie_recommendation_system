"""
Main Recommendation Engine

This module combines all components to provide a unified recommendation system
that uses both collaborative filtering and machine learning approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pickle
import json

class MovieRecommendationEngine:
    """
    Main recommendation engine that combines collaborative filtering and ML models.
    """
    
    def __init__(self, data_path: str = "data"):
        """
        Initialize the recommendation engine.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        
        # Data storage
        self.user_item_matrix = None
        self.movies_df = None
        self.ratings_df = None
        
        # Similarity matrices
        self.movie_similarity_matrix = None
        self.user_similarity_matrix = None
        
        # ML models
        self.ml_models = {}
        self.feature_scaler = None
        self.feature_columns = []
        
        # Statistics
        self.global_mean = 3.5
        self.user_means = {}
        self.movie_means = {}
        
        print("Movie Recommendation Engine initialized")
    
    def load_data(self):
        """Load all necessary data and models."""
        print("Loading data and models...")
        
        try:
            # Load processed data
            processed_path = self.data_path / "processed"
            
            if (processed_path / "processed_ratings.csv").exists():
                self.ratings_df = pd.read_csv(processed_path / "processed_ratings.csv")
                print(f"Loaded {len(self.ratings_df):,} ratings")
            
            # Load movies data
            raw_path = self.data_path / "raw"
            if (raw_path / "movies.csv").exists():
                self.movies_df = pd.read_csv(raw_path / "movies.csv")
                print(f"Loaded {len(self.movies_df):,} movies")
            
            # Load similarity matrices
            models_path = Path("models")
            
            if (models_path / "movie_similarity_matrix.pkl").exists():
                with open(models_path / "movie_similarity_matrix.pkl", "rb") as f:
                    self.movie_similarity_matrix = pickle.load(f)
                print("Loaded movie similarity matrix")
            
            if (models_path / "user_similarity_matrix.pkl").exists():
                with open(models_path / "user_similarity_matrix.pkl", "rb") as f:
                    self.user_similarity_matrix = pickle.load(f)
                print("Loaded user similarity matrix")
            
            # Load ML models
            if (models_path / "model_metadata.json").exists():
                with open(models_path / "model_metadata.json", "r") as f:
                    metadata = json.load(f)
                    self.feature_columns = metadata.get('feature_columns', [])
            
            # Load feature scaler
            if (models_path / "feature_scaler.pkl").exists():
                with open(models_path / "feature_scaler.pkl", "rb") as f:
                    self.feature_scaler = pickle.load(f)
                print("Loaded feature scaler")
            
            # Load individual models
            model_files = {
                'XGBoost': 'xgboost_model.pkl',
                'Random Forest': 'random_forest_model.pkl',
                'LightGBM': 'lightgbm_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = models_path / filename
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        self.ml_models[model_name] = pickle.load(f)
                    print(f"Loaded {model_name} model")
            
            # Calculate basic statistics
            self._calculate_statistics()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Some components may not be available")
    
    def _calculate_statistics(self):
        """Calculate basic statistics from the data."""
        if self.ratings_df is not None:
            self.global_mean = self.ratings_df['rating'].mean()
            
            # User means
            user_stats = self.ratings_df.groupby('userId')['rating'].mean()
            self.user_means = user_stats.to_dict()
            
            # Movie means
            movie_stats = self.ratings_df.groupby('movieId')['rating'].mean()
            self.movie_means = movie_stats.to_dict()
            
            print(f"Calculated statistics - Global mean: {self.global_mean:.2f}")
    
    def get_movie_info(self, movie_id: int) -> Dict[str, any]:
        """Get movie information by ID."""
        if self.movies_df is None:
            return {'movieId': movie_id, 'title': f'Movie_{movie_id}', 'genres': 'Unknown'}
        
        movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(movie_info) == 0:
            return {'movieId': movie_id, 'title': f'Movie_{movie_id}', 'genres': 'Unknown'}
        
        return movie_info.iloc[0].to_dict()
    
    def recommend_movies_similarity(self, user_id: int, method: str = 'item_based', 
                                  n_recommendations: int = 10) -> List[Dict[str, any]]:
        """
        Recommend movies using similarity-based collaborative filtering.
        
        Args:
            user_id: Target user ID
            method: 'item_based' or 'user_based'
            n_recommendations: Number of recommendations
            
        Returns:
            List of movie recommendations with metadata
        """
        print(f"Generating {method} similarity recommendations for user {user_id}")
        
        if method == 'item_based' and self.movie_similarity_matrix is None:
            print("Movie similarity matrix not available")
            return self._get_popular_movies(n_recommendations)
        
        if method == 'user_based' and self.user_similarity_matrix is None:
            print("User similarity matrix not available")
            return self._get_popular_movies(n_recommendations)
        
        # This is a simplified implementation
        # In practice, you'd use the collaborative filtering module
        recommendations = []
        
        if method == 'item_based':
            # Get user's rated movies
            if self.ratings_df is not None:
                user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
                
                if len(user_ratings) == 0:
                    return self._get_popular_movies(n_recommendations)
                
                # For each rated movie, find similar movies
                similar_movies = {}
                
                for _, rating_row in user_ratings.iterrows():
                    movie_id = rating_row['movieId']
                    user_rating = rating_row['rating']
                    
                    if movie_id in self.movie_similarity_matrix.index:
                        similarities = self.movie_similarity_matrix.loc[movie_id]
                        
                        # Get top similar movies
                        top_similar = similarities.nlargest(20)
                        
                        for sim_movie_id, similarity in top_similar.items():
                            if sim_movie_id != movie_id and similarity > 0.1:
                                if sim_movie_id not in similar_movies:
                                    similar_movies[sim_movie_id] = 0
                                similar_movies[sim_movie_id] += similarity * user_rating
                
                # Sort by score and get top recommendations
                sorted_movies = sorted(similar_movies.items(), key=lambda x: x[1], reverse=True)
                
                for movie_id, score in sorted_movies[:n_recommendations]:
                    movie_info = self.get_movie_info(movie_id)
                    movie_info['predicted_rating'] = min(5.0, max(0.5, score / 5))
                    movie_info['recommendation_score'] = score
                    movie_info['method'] = 'item_based_similarity'
                    recommendations.append(movie_info)
        
        return recommendations if recommendations else self._get_popular_movies(n_recommendations)
    
    def recommend_movies_ml(self, user_id: int, model_name: str = 'XGBoost', 
                           n_recommendations: int = 10) -> List[Dict[str, any]]:
        """
        Recommend movies using machine learning models.
        
        Args:
            user_id: Target user ID
            model_name: Name of the ML model to use
            n_recommendations: Number of recommendations
            
        Returns:
            List of movie recommendations with metadata
        """
        print(f"Generating ML recommendations for user {user_id} using {model_name}")
        
        if model_name not in self.ml_models:
            print(f"Model {model_name} not available")
            return self._get_popular_movies(n_recommendations)
        
        model = self.ml_models[model_name]
        recommendations = []
        
        # Get user statistics
        user_mean = self.user_means.get(user_id, self.global_mean)
        
        # Get user's rated movies to exclude them
        rated_movies = set()
        if self.ratings_df is not None:
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            rated_movies = set(user_ratings['movieId'].tolist())
        
        # Generate predictions for unrated movies
        movie_scores = []
        
        # Sample from available movies (in practice, you'd predict for all movies)
        if self.movies_df is not None:
            available_movies = self.movies_df['movieId'].tolist()
        else:
            available_movies = list(range(1, 1000))  # Dummy range
        
        for movie_id in available_movies[:500]:  # Limit for performance
            if movie_id not in rated_movies:
                movie_mean = self.movie_means.get(movie_id, self.global_mean)
                
                # Create feature vector (simplified)
                features = {
                    'user_mean': user_mean,
                    'movie_mean': movie_mean,
                    'rating_gmean': self.global_mean,
                    'user_bias': user_mean - self.global_mean,
                    'movie_bias': movie_mean - self.global_mean,
                }
                
                # Add other features with defaults
                for feature in self.feature_columns:
                    if feature not in features:
                        features[feature] = 0.0
                
                try:
                    predicted_rating = self._predict_with_ml_model(model, features, model_name)
                    movie_scores.append((movie_id, predicted_rating))
                except Exception as e:
                    print(f"Error predicting for movie {movie_id}: {e}")
                    continue
        
        # Sort by predicted rating and get top recommendations
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        for movie_id, predicted_rating in movie_scores[:n_recommendations]:
            movie_info = self.get_movie_info(movie_id)
            movie_info['predicted_rating'] = predicted_rating
            movie_info['method'] = f'ml_{model_name.lower()}'
            recommendations.append(movie_info)
        
        return recommendations if recommendations else self._get_popular_movies(n_recommendations)
    
    def _predict_with_ml_model(self, model, features: Dict[str, float], model_name: str) -> float:
        """Make prediction using ML model."""
        # Create feature vector
        feature_vector = np.zeros(len(self.feature_columns))
        for i, feature in enumerate(self.feature_columns):
            if feature in features:
                feature_vector[i] = features[feature]
        
        # Scale if needed (for linear models)
        if model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            if self.feature_scaler is not None:
                feature_vector = self.feature_scaler.transform(feature_vector.reshape(1, -1))
                prediction = model.predict(feature_vector)[0]
            else:
                prediction = model.predict(feature_vector.reshape(1, -1))[0]
        else:
            prediction = model.predict(feature_vector.reshape(1, -1))[0]
        
        # Clamp to valid rating range
        return max(0.5, min(5.0, prediction))
    
    def _get_popular_movies(self, n_recommendations: int) -> List[Dict[str, any]]:
        """Get popular movies as fallback recommendations."""
        print("Using popular movies as fallback recommendations")
        
        recommendations = []
        
        if self.ratings_df is not None:
            # Get most rated movies
            movie_counts = self.ratings_df['movieId'].value_counts()
            popular_movies = movie_counts.head(n_recommendations)
            
            for movie_id, count in popular_movies.items():
                movie_info = self.get_movie_info(movie_id)
                movie_info['predicted_rating'] = self.movie_means.get(movie_id, self.global_mean)
                movie_info['popularity_score'] = count
                movie_info['method'] = 'popularity'
                recommendations.append(movie_info)
        else:
            # Dummy popular movies
            for i in range(1, n_recommendations + 1):
                movie_info = {
                    'movieId': i,
                    'title': f'Popular Movie {i}',
                    'genres': 'Unknown',
                    'predicted_rating': 4.0,
                    'method': 'popularity_fallback'
                }
                recommendations.append(movie_info)
        
        return recommendations
    
    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 10,
                                 weights: Dict[str, float] = None) -> List[Dict[str, any]]:
        """
        Get hybrid recommendations combining multiple methods.
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            weights: Weights for different methods
            
        Returns:
            List of hybrid recommendations
        """
        if weights is None:
            weights = {
                'item_based': 0.3,
                'user_based': 0.2,
                'ml_xgboost': 0.4,
                'popularity': 0.1
            }
        
        print(f"Generating hybrid recommendations for user {user_id}")
        
        all_recommendations = {}
        
        # Get recommendations from different methods
        methods = [
            ('item_based', lambda: self.recommend_movies_similarity(user_id, 'item_based', n_recommendations * 2)),
            ('ml_xgboost', lambda: self.recommend_movies_ml(user_id, 'XGBoost', n_recommendations * 2)),
            ('popularity', lambda: self._get_popular_movies(n_recommendations))
        ]
        
        for method_name, get_recs_func in methods:
            if method_name in weights and weights[method_name] > 0:
                try:
                    recs = get_recs_func()
                    for rec in recs:
                        movie_id = rec['movieId']
                        if movie_id not in all_recommendations:
                            all_recommendations[movie_id] = {
                                'movie_info': rec,
                                'scores': {},
                                'total_score': 0
                            }
                        
                        score = rec.get('predicted_rating', 3.5) * weights[method_name]
                        all_recommendations[movie_id]['scores'][method_name] = score
                        all_recommendations[movie_id]['total_score'] += score
                except Exception as e:
                    print(f"Error getting {method_name} recommendations: {e}")
        
        # Sort by total score
        sorted_recs = sorted(all_recommendations.items(), 
                           key=lambda x: x[1]['total_score'], reverse=True)
        
        # Prepare final recommendations
        final_recommendations = []
        for movie_id, rec_data in sorted_recs[:n_recommendations]:
            movie_info = rec_data['movie_info'].copy()
            movie_info['hybrid_score'] = rec_data['total_score']
            movie_info['method_scores'] = rec_data['scores']
            movie_info['method'] = 'hybrid'
            final_recommendations.append(movie_info)
        
        return final_recommendations
    
    def predict_user_rating(self, user_id: int, movie_id: int, 
                           method: str = 'ml') -> Dict[str, any]:
        """
        Predict what rating a user would give to a specific movie.
        
        Args:
            user_id: Target user ID
            movie_id: Target movie ID
            method: Prediction method ('ml', 'item_based', 'user_based')
            
        Returns:
            Dictionary with prediction details
        """
        result = {
            'user_id': user_id,
            'movie_id': movie_id,
            'method': method,
            'predicted_rating': self.global_mean,
            'confidence': 0.5
        }
        
        if method == 'ml' and 'XGBoost' in self.ml_models:
            user_mean = self.user_means.get(user_id, self.global_mean)
            movie_mean = self.movie_means.get(movie_id, self.global_mean)
            
            features = {
                'user_mean': user_mean,
                'movie_mean': movie_mean,
                'rating_gmean': self.global_mean,
                'user_bias': user_mean - self.global_mean,
                'movie_bias': movie_mean - self.global_mean,
            }
            
            # Add other features with defaults
            for feature in self.feature_columns:
                if feature not in features:
                    features[feature] = 0.0
            
            try:
                predicted_rating = self._predict_with_ml_model(
                    self.ml_models['XGBoost'], features, 'XGBoost'
                )
                result['predicted_rating'] = predicted_rating
                result['confidence'] = 0.8
            except Exception as e:
                print(f"Error in ML prediction: {e}")
        
        # Add movie information
        movie_info = self.get_movie_info(movie_id)
        result.update(movie_info)
        
        return result

def main():
    """
    Example usage of the MovieRecommendationEngine.
    """
    print("Movie Recommendation Engine Demo")
    
    # Initialize engine
    engine = MovieRecommendationEngine()
    
    # Try to load data (will handle missing files gracefully)
    engine.load_data()
    
    # Test user ID
    test_user_id = 123
    
    print(f"\n=== Recommendations for User {test_user_id} ===")
    
    # Try different recommendation methods
    try:
        # Similarity-based recommendations
        print("\n1. Item-based Collaborative Filtering:")
        item_recs = engine.recommend_movies_similarity(test_user_id, 'item_based', 5)
        for i, rec in enumerate(item_recs, 1):
            print(f"   {i}. {rec['title']} - Rating: {rec.get('predicted_rating', 'N/A'):.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    try:
        # ML-based recommendations
        print("\n2. Machine Learning (XGBoost):")
        ml_recs = engine.recommend_movies_ml(test_user_id, 'XGBoost', 5)
        for i, rec in enumerate(ml_recs, 1):
            print(f"   {i}. {rec['title']} - Rating: {rec.get('predicted_rating', 'N/A'):.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    try:
        # Hybrid recommendations
        print("\n3. Hybrid Recommendations:")
        hybrid_recs = engine.get_hybrid_recommendations(test_user_id, 5)
        for i, rec in enumerate(hybrid_recs, 1):
            print(f"   {i}. {rec['title']} - Score: {rec.get('hybrid_score', 'N/A'):.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test rating prediction
    print(f"\n=== Rating Prediction ===")
    test_movie_id = 1
    prediction = engine.predict_user_rating(test_user_id, test_movie_id)
    print(f"User {test_user_id} would rate '{prediction['title']}': {prediction['predicted_rating']:.2f}")
    
    print("\n=== Demo Complete ===")
    print("Note: Full functionality requires processed data and trained models.")

if __name__ == "__main__":
    main()
