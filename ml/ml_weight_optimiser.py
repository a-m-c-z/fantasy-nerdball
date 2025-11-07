#!/usr/bin/env python3
"""
Train ML model to optimise weighting parameters for FPL score calculation.
Uses gradient boosting to learn optimal weights for form, historical 
performance, and fixture difficulty.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json


class FPLWeightOptimiser:
    """Optimises scoring weights using machine learning."""
    
    def __init__(self, training_data_path):
        """
        Initialise the weight optimiser.
        
        Args:
            training_data_path (str): Path to training data directory
        """
        self.training_data_path = Path(training_data_path)
        self.models = {}  # Store models by position
        self.optimal_weights = {}
        
    def load_training_data(self):
        """Load all training data and aggregate by position."""
        print("\n=== Loading Training Data ===")
        
        all_data = []
        player_files = list(self.training_data_path.glob("*.csv"))
        
        print(f"Found {len(player_files)} player files")
        
        for player_file in player_files:
            try:
                df = pd.read_csv(player_file)
                
                # Add player identifier
                df['player'] = player_file.stem
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error loading {player_file}: {e}")
        
        if not all_data:
            raise ValueError("No training data loaded!")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"Loaded {len(combined_df)} gameweek records")
        print(f"Features: {list(combined_df.columns)}")
        
        return combined_df
    
    def prepare_features(self, df):
        """
        Prepare features for model training.
        
        Args:
            df (pd.DataFrame): Combined training data
            
        Returns:
            tuple: (features_df, target_series)
        """
        print("\n=== Preparing Features ===")
        
        # Features we'll use for prediction
        feature_columns = [
            'form', 'historical_ppg', 'fixture_difficulty',
            'minutes', 'creativity', 'ict_index', 'influence', 'threat'
        ]
        
        # Ensure all feature columns exist
        available_features = [
            col for col in feature_columns if col in df.columns
        ]
        
        print(f"Using features: {available_features}")
        
        # Target variable
        target = 'total_points'
        
        # Remove rows with missing values
        df_clean = df[available_features + [target]].dropna()
        
        print(f"Clean dataset size: {len(df_clean)} rows")
        
        X = df_clean[available_features]
        y = df_clean[target]
        
        return X, y
    
    def train_model(self, X, y, position=None):
        """
        Train gradient boosting model.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            position (str): Position identifier (optional)
            
        Returns:
            GradientBoostingRegressor: Trained model
        """
        position_label = position if position else "All"
        print(f"\n=== Training Model for {position_label} ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\nTraining Performance:")
        print(f"  MAE: {train_mae:.3f}")
        print(f"  RMSE: {train_rmse:.3f}")
        print(f"  R²: {train_r2:.3f}")
        
        print(f"\nTest Performance:")
        print(f"  MAE: {test_mae:.3f}")
        print(f"  RMSE: {test_rmse:.3f}")
        print(f"  R²: {test_r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return model, feature_importance
    
    def extract_optimal_weights(self, feature_importance):
        """
        Extract optimal weights from feature importance.
        
        Args:
            feature_importance (pd.DataFrame): Feature importance scores
            
        Returns:
            dict: Optimal weights for form, historic, difficulty
        """
        # Get importance scores for key features
        form_importance = feature_importance[
            feature_importance['feature'] == 'form'
        ]['importance'].values
        
        hist_importance = feature_importance[
            feature_importance['feature'] == 'historical_ppg'
        ]['importance'].values
        
        diff_importance = feature_importance[
            feature_importance['feature'] == 'fixture_difficulty'
        ]['importance'].values
        
        # Extract values or use 0 if not found
        form_imp = form_importance[0] if len(form_importance) > 0 else 0
        hist_imp = hist_importance[0] if len(hist_importance) > 0 else 0
        diff_imp = diff_importance[0] if len(diff_importance) > 0 else 0
        
        # Normalise to sum to 1.0
        total = form_imp + hist_imp + diff_imp
        
        if total == 0:
            # Fallback to equal weights
            return {
                'form': 0.33,
                'historic': 0.33,
                'difficulty': 0.34
            }
        
        weights = {
            'form': round(form_imp / total, 3),
            'historic': round(hist_imp / total, 3),
            'difficulty': round(diff_imp / total, 3)
        }
        
        # Ensure weights sum to 1.0 (rounding adjustment)
        weight_sum = sum(weights.values())
        if weight_sum != 1.0:
            diff = 1.0 - weight_sum
            weights['form'] = round(weights['form'] + diff, 3)
        
        return weights
    
    def train_all_positions(self, df):
        """
        Train separate models for each position if position data available.
        
        Args:
            df (pd.DataFrame): Training data
            
        Returns:
            dict: Optimal weights by position
        """
        # For now, train single model across all positions
        # Position-specific training would require position labels
        print("\n=== Training Combined Model (All Positions) ===")
        
        X, y = self.prepare_features(df)
        model, feature_importance = self.train_model(X, y, position=None)
        
        # Extract optimal weights
        optimal_weights = self.extract_optimal_weights(feature_importance)
        
        print(f"\n=== Optimal Weights (Combined) ===")
        print(f"Form: {optimal_weights['form']:.3f}")
        print(f"Historic: {optimal_weights['historic']:.3f}")
        print(f"Difficulty: {optimal_weights['difficulty']:.3f}")
        
        return {
            'ALL': optimal_weights
        }
    
    def save_optimal_weights(self, weights, output_path):
        """
        Save optimal weights to JSON file.
        
        Args:
            weights (dict): Optimal weights by position
            output_path (str): Path to save weights
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(weights, f, indent=2)
        
        print(f"\n✅ Optimal weights saved to: {output_file}")
    
    def generate_config_snippet(self, weights):
        """
        Generate config snippet with optimal weights.
        
        Args:
            weights (dict): Optimal weights by position
        """
        print("\n=== Suggested Config Updates ===")
        print("\nAdd to config.py:")
        print("```python")
        
        # If we have position-specific weights, show those
        if len(weights) > 1 or 'ALL' not in weights:
            print("POSITION_SCORING_WEIGHTS = {")
            for position, pos_weights in weights.items():
                print(f'    "{position}": {{')
                print(f'        "form": {pos_weights["form"]:.3f},')
                print(f'        "historic": {pos_weights["historic"]:.3f},')
                print(f'        "difficulty": {pos_weights["difficulty"]:.3f}')
                print('    },')
            print("}")
        else:
            # Single set of weights for all positions
            w = weights['ALL']
            print("# Recommended for all positions:")
            print("POSITION_SCORING_WEIGHTS = {")
            for pos in ["GK", "DEF", "MID", "FWD"]:
                print(f'    "{pos}": {{')
                print(f'        "form": {w["form"]:.3f},')
                print(f'        "historic": {w["historic"]:.3f},')
                print(f'        "difficulty": {w["difficulty"]:.3f}')
                print('    },')
            print("}")
        
        print("```")


def main():
    """Main function to train model and optimise weights."""
    # Path to training data
    training_data_path = "ml/training_data/2019-20"
    
    print("=== FPL Weight Optimisation with Machine Learning ===")
    print(f"Training data: {training_data_path}")
    
    # Initialise optimiser
    optimiser = FPLWeightOptimiser(training_data_path)
    
    # Load training data
    df = optimiser.load_training_data()
    
    # Train model(s)
    optimal_weights = optimiser.train_all_positions(df)
    
    # Save optimal weights
    output_path = "ml/optimal_weights_2019-20.json"
    optimiser.save_optimal_weights(optimal_weights, output_path)
    
    # Generate config snippet
    optimiser.generate_config_snippet(optimal_weights)
    
    print("\n✅ Weight optimisation complete!")


if __name__ == "__main__":
    main()