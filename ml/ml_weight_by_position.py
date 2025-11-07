#!/usr/bin/env python3
"""
Multi-season weight optimisation comparison by position.
Trains position-specific models on multiple seasons to find optimal weights 
for each position.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from ml_weight_optimiser import FPLWeightOptimiser


class MultiSeasonWeightComparison:
    """Compare optimal weights across multiple seasons by position."""
    
    def __init__(self):
        """Initialise multi-season comparison."""
        self.results = []
    
    def train_season_model(self, season):
        """
        Train position-specific models for a season.
        
        Args:
            season (str): Season to train (e.g. '2019-20')
            
        Returns:
            dict: Results including weights by position
        """
        print(f"\n{'='*60}")
        print(f"Training models for {season}")
        print(f"{'='*60}")
        
        training_data_path = f"ml/training_data/{season}"
        
        if not Path(training_data_path).exists():
            print(f"✗ Training data not found: {training_data_path}")
            return None
        
        try:
            optimiser = FPLWeightOptimiser(training_data_path)
            
            # Load training data
            df = optimiser.load_training_data()
            
            # Check if position column exists
            if 'position' not in df.columns:
                print(f"✗ Position column not found in training data")
                return None
            
            # Train models for each position
            position_results = {}
            
            for position in ['GK', 'DEF', 'MID', 'FWD']:
                print(f"\n  Training {position} model...")
                
                # Filter to this position
                position_df = df[df['position'] == position].copy()
                
                if len(position_df) < 100:
                    print(f"  ⚠ Insufficient data for {position} "
                          f"({len(position_df)} samples), skipping")
                    continue
                
                # Prepare features and train
                X, y = optimiser.prepare_features(position_df)
                model, feature_importance = optimiser.train_model(
                    X, y, position=f"{season}-{position}"
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
                
                position_results[position] = {
                    'weights': optimal_weights,
                    'cv_mae': -cv_scores.mean(),
                    'cv_mae_std': cv_scores.std(),
                    'n_samples': len(position_df)
                }
                
                print(f"    Form: {optimal_weights['form']:.3f}, "
                      f"Historic: {optimal_weights['historic']:.3f}, "
                      f"Difficulty: {optimal_weights['difficulty']:.3f}")
                print(f"    CV MAE: {-cv_scores.mean():.3f}")
                print(f"    Samples: {len(position_df)}")
            
            results = {
                'season': season,
                'positions': position_results,
                'total_samples': len(df)
            }
            
            print(f"\n✓ Training complete for {season}")
            
            return results
            
        except Exception as e:
            print(f"✗ Error training model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_seasons(self, seasons_to_process):
        """
        Compare optimal weights across multiple seasons.
        
        Args:
            seasons_to_process (list): List of season strings
        """
        print("\n" + "="*60)
        print("MULTI-SEASON POSITION-SPECIFIC WEIGHT OPTIMISATION")
        print("="*60)
        
        for season in seasons_to_process:
            # Train model for this season
            results = self.train_season_model(season)
            
            if results:
                self.results.append(results)
        
        if not self.results:
            print("\n✗ No successful results to compare")
            return
        
        # Analyse and compare results
        self._analyse_results()
    
    def _analyse_results(self):
        """Analyse and display comparison of results by position."""
        print("\n" + "="*60)
        print("COMPARISON RESULTS BY POSITION")
        print("="*60)
        
        # Collect results for each position across seasons
        position_data = {
            'GK': [],
            'DEF': [],
            'MID': [],
            'FWD': []
        }
        
        for result in self.results:
            season = result['season']
            for position, pos_result in result['positions'].items():
                position_data[position].append({
                    'Season': season,
                    'Form': pos_result['weights']['form'],
                    'Historic': pos_result['weights']['historic'],
                    'Difficulty': pos_result['weights']['difficulty'],
                    'CV MAE': pos_result['cv_mae'],
                    'Samples': pos_result['n_samples']
                })
        
        # Display results for each position
        all_position_averages = {}
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            if not position_data[position]:
                print(f"\n⚠ No data for {position}")
                continue
            
            print(f"\n{'='*60}")
            print(f"{position} POSITION")
            print(f"{'='*60}")
            
            pos_df = pd.DataFrame(position_data[position])
            print("\n" + pos_df.to_string(index=False))
            
            # Calculate averages
            avg_weights = {
                'form': pos_df['Form'].mean(),
                'historic': pos_df['Historic'].mean(),
                'difficulty': pos_df['Difficulty'].mean()
            }
            
            std_weights = {
                'form': pos_df['Form'].std(),
                'historic': pos_df['Historic'].std(),
                'difficulty': pos_df['Difficulty'].std()
            }
            
            all_position_averages[position] = {
                'avg': avg_weights,
                'std': std_weights
            }
            
            print(f"\nAverage Weights for {position}:")
            print(f"  Form: {avg_weights['form']:.3f} "
                  f"(±{std_weights['form']:.3f})")
            print(f"  Historic: {avg_weights['historic']:.3f} "
                  f"(±{std_weights['historic']:.3f})")
            print(f"  Difficulty: {avg_weights['difficulty']:.3f} "
                  f"(±{std_weights['difficulty']:.3f})")
        
        # Generate recommended config
        self._generate_recommended_config(all_position_averages)
    
    def _generate_recommended_config(self, position_averages):
        """
        Generate recommended config based on position-specific analysis.
        
        Args:
            position_averages (dict): Average weights by position
        """
        print("\n" + "="*60)
        print("RECOMMENDED CONFIG.PY UPDATES")
        print("="*60)
        
        print("\nBased on multi-season, position-specific analysis:")
        print("\n```python")
        print("# ML-Optimised Weights (Position-Specific)")
        print("POSITION_SCORING_WEIGHTS = {")
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            if position not in position_averages:
                continue
                
            avg = position_averages[position]['avg']
            std = position_averages[position]['std']
            
            print(f'    "{position}": {{')
            print(f'        "form": {avg["form"]:.3f},  '
                  f'# ±{std["form"]:.3f}')
            print(f'        "historic": {avg["historic"]:.3f},  '
                  f'# ±{std["historic"]:.3f}')
            print(f'        "difficulty": {avg["difficulty"]:.3f}  '
                  f'# ±{std["difficulty"]:.3f}')
            print('    },')
        
        print("}")
        print("```")
        
        # Provide interpretation
        print("\n=== Interpretation ===")
        
        # Compare positions
        print("\nPosition Comparisons:")
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            if position not in position_averages:
                continue
            
            avg = position_averages[position]['avg']
            std = position_averages[position]['std']
            
            consistency = "✓" if all(s < 0.1 for s in std.values()) else "⚠"
            print(f"\n{position}: {consistency}")
            print(f"  Form dominance: {avg['form']:.1%}")
            print(f"  Historical weight: {avg['historic']:.1%}")
            print(f"  Fixture sensitivity: {avg['difficulty']:.1%}")


def main():
    """Main function for multi-season position-specific comparison."""
    # Seasons to process
    seasons_to_process = [
        "2019-20", "2020-21", "2021-22",
        "2022-23", "2023-24", "2024-25"
    ]
    
    comparison = MultiSeasonWeightComparison()
    comparison.compare_seasons(seasons_to_process)
    
    print("\n✅ Multi-season position-specific comparison complete!")


if __name__ == "__main__":
    main()