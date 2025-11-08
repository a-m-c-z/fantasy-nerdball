#!/usr/bin/env python3
"""
Multi-season weight optimisation comparison by position.
Trains position-specific models on multiple seasons to find optimal weights 
for each position, now including per-gameweek dynamic weighting and 
visualisation.
"""

import sys
import pandas as pd
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
        output_dir = Path("ml_output/raw_output")
        output_dir.mkdir(exist_ok=True)

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

            # Compute per-gameweek dynamic weights and plots
            optimiser.compute_dynamic_weights(df, output_dir)

            print(f"✓ Dynamic weighting completed for {season}")
            return optimiser.optimal_weights

        except Exception as e:
            print(f"✗ Error processing {season}: {e}")
            return None


if __name__ == "__main__":
    # Run for all provided seasons
    comparison = MultiSeasonWeightComparison()
    seasons = sys.argv[1:] if len(sys.argv) > 1 else ["2024-25"]
    for s in seasons:
        comparison.train_season_model(s)
