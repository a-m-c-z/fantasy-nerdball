#!/usr/bin/env python3
"""
Train ML model to optimise weighting parameters for FPL score calculation.
Uses gradient boosting to learn optimal weights for form, historical 
performance, and fixture difficulty. Now supports dynamic per-gameweek
weighting and output visualisation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import matplotlib.pyplot as plt
import os


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
                df["player"] = player_file.stem
                all_data.append(df)
            except Exception as e:
                print(f"⚠️  Failed to load {player_file.name}: {e}")

        if not all_data:
            raise ValueError("No valid training data files found.")

        combined = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined)} total records.")
        return combined


    def optimise_weights(self, df):
        """
        Compute optimal feature importances for each position using
        gradient boosting.
        """
        positions = df["position"].unique()
        results = {}

        for pos in positions:
            print(f"\n🔍 Optimising weights for {pos}")
            df_pos = df[df["position"] == pos].copy()

            X = df_pos[["form", "historical_ppg", "fixture_difficulty"]]
            y = df_pos["total_points"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = GradientBoostingRegressor(random_state=42)
            model.fit(X_train, y_train)
            self.models[pos] = model

            importances = model.feature_importances_
            norm_weights = 100 * importances / importances.sum()

            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            results[pos] = {
                "form_dominance": norm_weights[0],
                "historical_weight": norm_weights[1],
                "fixture_sensitivity": norm_weights[2],
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }

            print(f"{pos}: ✓")
            print(f"  Form dominance: {norm_weights[0]:.1f}%")
            print(f"  Historical weight: {norm_weights[1]:.1f}%")
            print(f"  Fixture sensitivity: {norm_weights[2]:.1f}%")
            print(f"  MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

        self.optimal_weights = results
        return results


    def compute_dynamic_weights(self, df, output_dir):
        """
        Compute optimal weighting for each gameweek and position.
        Outputs one CSV and one stacked bar chart per position.
        """
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        df = df.copy()
        if "gameweek" not in df.columns:
            raise ValueError("Expected 'gameweek' column missing from training"
                             " data.")

        # Shift form to use previous GW
        df["form_lagged"] = df.groupby("player")["form"].shift(1)
        df["form_lagged"] = df["form_lagged"].fillna(0)

        FEATURES = ["form_lagged", "historical_ppg", "fixture_difficulty"]
        TARGET = "total_points"
        positions = df["position"].unique()

        for pos in positions:
            print(f"\n📊 Computing dynamic weights for {pos}...")
            df_pos = df[df["position"] == pos].copy()
            weights_over_time = []

            for gw, gw_data in df_pos.groupby("gameweek"):
                if len(gw_data) < 5:
                    continue

                X = gw_data[FEATURES]
                y = gw_data[TARGET]

                model = GradientBoostingRegressor(random_state=42)
                model.fit(X, y)

                importances = model.feature_importances_
                norm_weights = 100 * importances / importances.sum()

                weights_over_time.append({
                    "gameweek": gw,
                    "form_dominance": norm_weights[0],
                    "historical_weight": norm_weights[1],
                    "fixture_sensitivity": norm_weights[2]
                })

            df_weights = pd.DataFrame(weights_over_time)
            csv_path = output_dir / f"weights_{pos}.csv"
            df_weights.to_csv(csv_path, index=False)
            self.optimal_weights[pos] = df_weights.iloc[-1].to_dict()

            # Plot
            plt.figure(figsize=(10, 6))
            plt.bar(
                df_weights["gameweek"],
                df_weights["form_dominance"],
                label="Form Dominance"
                )
            plt.bar(
                df_weights["gameweek"],
                df_weights["historical_weight"],
                bottom=df_weights["form_dominance"],
                label="Historical Weight"
                )
            bottom_sum = (
                df_weights["form_dominance"] + df_weights["historical_weight"]
            )
            plt.bar(df_weights["gameweek"], df_weights["fixture_sensitivity"],
                    bottom=bottom_sum, label="Fixture Sensitivity")

            plt.title(f"Optimal Weighting by Gameweek — {pos}")
            plt.xlabel("Gameweek")
            plt.ylabel("Weight (%)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"weights_{pos}.png")
            plt.close()

            print(f"✅ Saved {pos} weights to {csv_path}")

        print(f"\nAll dynamic weights saved to: {output_dir.resolve()}")
