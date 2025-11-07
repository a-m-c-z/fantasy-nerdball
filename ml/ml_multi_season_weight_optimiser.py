#!/usr/bin/env python3
"""
Multi-season weight optimisation comparison.
Trains models on multiple seasons and compares optimal weights to find 
consistent patterns.
"""

import sys
import subprocess
import pandas as pd
import json
from pathlib import Path
from ml_weight_optimiser import FPLWeightOptimiser


class MultiSeasonWeightComparison:
    """Compare optimal weights across multiple seasons."""
    
    def __init__(self):
        """Initialise multi-season comparison."""
        self.results = []
    
    def prepare_season_data(self, target_season, historical_seasons):
        """
        Prepare training data for a specific season.
        
        Args:
            target_season (str): Target season (e.g. '2019-20')
            historical_seasons (list): Previous seasons for historical data
            
        Returns:
            bool: True if successful
        """
        print(f"\n{'='*60}")
        print(f"Preparing data for {target_season}")
        print(f"{'='*60}")
        
        try:
            # Modify and run prepare_ml_training_data.py
            # For simplicity, we'll just run it with subprocess
            result = subprocess.run(
                [sys.executable, 'prepare_ml_training_data.py'],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                print("✓ Data preparation successful")
                return True
            else:
                print(f"✗ Data preparation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ Error preparing data: {e}")
            return False
    
    def train_season_model(self, season):
        """
        Train model for a specific season.
        
        Args:
            season (str): Season to train (e.g. '2019-20')
            
        Returns:
            dict: Results including weights and metrics
        """
        print(f"\n{'='*60}")
        print(f"Training model for {season}")
        print(f"{'='*60}")
        
        training_data_path = f"ml/training_data/{season}"
        
        if not Path(training_data_path).exists():
            print(f"✗ Training data not found: {training_data_path}")
            return None
        
        try:
            optimiser = FPLWeightOptimiser(training_data_path)
            
            # Load training data
            df = optimiser.load_training_data()
            
            # Train model
            X, y = optimiser.prepare_features(df)
            model, feature_importance = optimiser.train_model(
                X, y, position=season
            )
            
            # Extract optimal weights
            optimal_weights = optimiser.extract_optimal_weights(
                feature_importance
            )
            
            # Calculate validation metrics
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(
                model, X, y, cv=5, 
                scoring='neg_mean_absolute_error'
            )
            
            results = {
                'season': season,
                'weights': optimal_weights,
                'feature_importance': feature_importance.to_dict('records'),
                'cv_mae': -cv_scores.mean(),
                'cv_mae_std': cv_scores.std(),
                'n_samples': len(df)
            }
            
            print(f"\n✓ Training complete for {season}")
            print(f"  Optimal weights: {optimal_weights}")
            print(f"  CV MAE: {results['cv_mae']:.3f} "
                  f"(±{results['cv_mae_std']:.3f})")
            
            return results
            
        except Exception as e:
            print(f"✗ Error training model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_seasons(self, seasons_config):
        """
        Compare optimal weights across multiple seasons.
        
        Args:
            seasons_config (list): List of (target, historical) tuples
        """
        print("\n" + "="*60)
        print("MULTI-SEASON WEIGHT OPTIMISATION COMPARISON")
        print("="*60)
        
        for target_season, historical_seasons in seasons_config:
            # Train model for this season
            results = self.train_season_model(target_season)
            
            if results:
                self.results.append(results)
        
        if not self.results:
            print("\n✗ No successful results to compare")
            return
        
        # Analyse and compare results
        self._analyse_results()
    
    def _analyse_results(self):
        """Analyse and display comparison of results."""
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        # Create comparison dataframe
        comparison_data = []
        for result in self.results:
            row = {
                'Season': result['season'],
                'Form Weight': result['weights']['form'],
                'Historic Weight': result['weights']['historic'],
                'Difficulty Weight': result['weights']['difficulty'],
                'CV MAE': result['cv_mae'],
                'Samples': result['n_samples']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n=== Weight Comparison by Season ===")
        print(comparison_df.to_string(index=False))
        
        # Calculate average weights
        avg_weights = {
            'form': comparison_df['Form Weight'].mean(),
            'historic': comparison_df['Historic Weight'].mean(),
            'difficulty': comparison_df['Difficulty Weight'].mean()
        }
        
        std_weights = {
            'form': comparison_df['Form Weight'].std(),
            'historic': comparison_df['Historic Weight'].std(),
            'difficulty': comparison_df['Difficulty Weight'].std()
        }
        
        print("\n=== Average Optimal Weights (Across Seasons) ===")
        print(f"Form: {avg_weights['form']:.3f} "
              f"(±{std_weights['form']:.3f})")
        print(f"Historic: {avg_weights['historic']:.3f} "
              f"(±{std_weights['historic']:.3f})")
        print(f"Difficulty: {avg_weights['difficulty']:.3f} "
              f"(±{std_weights['difficulty']:.3f})")
        
        # Generate recommended config
        self._generate_recommended_config(avg_weights, std_weights)
    
    def _generate_recommended_config(self, avg_weights, std_weights):
        """
        Generate recommended config based on multi-season analysis.
        
        Args:
            avg_weights (dict): Average weights across seasons
            std_weights (dict): Standard deviation of weights
        """
        print("\n" + "="*60)
        print("RECOMMENDED CONFIG.PY UPDATES")
        print("="*60)
        
        print("\nBased on multi-season analysis:")
        print("\n```python")
        print("# ML-Optimised Weights (Multi-Season Average)")
        print("POSITION_SCORING_WEIGHTS = {")
        
        for pos in ["GK", "DEF", "MID", "FWD"]:
            print(f'    "{pos}": {{')
            print(f'        "form": {avg_weights["form"]:.3f},  '
                  f'# ±{std_weights["form"]:.3f}')
            print(f'        "historic": {avg_weights["historic"]:.3f},  '
                  f'# ±{std_weights["historic"]:.3f}')
            print(f'        "difficulty": {avg_weights["difficulty"]:.3f}  '
                  f'# ±{std_weights["difficulty"]:.3f}')
            print('    },')
        
        print("}")
        print("```")
        
        # Provide interpretation
        print("\n=== Interpretation ===")
        
        if std_weights['form'] > 0.1:
            print("⚠ Form weight varies significantly across seasons")
            print("  Consider using season-specific weights")
        else:
            print("✓ Form weight is consistent across seasons")
        
        if std_weights['historic'] > 0.1:
            print("⚠ Historic weight varies significantly across seasons")
        else:
            print("✓ Historic weight is consistent across seasons")
        
        if std_weights['difficulty'] > 0.1:
            print("⚠ Difficulty weight varies significantly across seasons")
        else:
            print("✓ Difficulty weight is consistent across seasons")


def main():
    """Main function for multi-season comparison."""
    # Define seasons to compare
    # Format: (target_season, [historical_seasons])
    seasons_config = [
        ("2019-20", ["2016-17", "2017-18", "2018-19"]),
        ("2020-21", ["2017-18", "2018-19", "2019-20"]),
        ("2021-22", ["2018-19", "2019-20", "2020-21"]),
        ("2022-23", ["2019-20", "2020-21", "2021-22"]),
        ("2023-24", ["2019-20", "2020-21", "2022-23"]),
        ("2024-25", ["2020-21", "2022-23", "2023-24"]),
    ]
    
    comparison = MultiSeasonWeightComparison()
    comparison.compare_seasons(seasons_config)
    
    print("\n✅ Multi-season comparison complete!")


if __name__ == "__main__":
    main()