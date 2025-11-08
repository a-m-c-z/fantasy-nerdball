"""
ML Weight Loader Utility
Handles loading of ML-optimised weights from CSV files.
"""

import os
import pandas as pd


class MLWeightLoader:
    """Handles loading and processing of ML-optimised weights from CSVs."""
    
    def __init__(self, config):
        """
        Initialise the ML weight loader.
        
        Args:
            config: Config object with settings like GAMEWEEK and 
                   GRANULAR_OUTPUT
        """
        self.config = config
        self.base_path = "ml_output/linear_regressions"
    
    def load_weights_for_position(self, position):
        """
        Load ML weights for a specific position from CSV file.
        
        Args:
            position (str): Position code (GK, DEF, MID, or FWD)
            
        Returns:
            dict: Dictionary with 'form', 'historic', and 'difficulty' keys,
                  or None if loading failed
        """
        csv_path = os.path.join(
            self.base_path, 
            f"smoothed_weights_{position}.csv"
        )
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Validate required columns exist
            required_cols = [
                'gameweek', 
                'form_dominance_smoothed', 
                'historical_weight_smoothed',
                'fixture_sensitivity_smoothed'
            ]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(
                    f"Missing required columns in {csv_path}"
                )
            
            # Filter for the current gameweek
            gw_data = df[df['gameweek'] == self.config.GAMEWEEK]
            
            if gw_data.empty:
                raise ValueError(
                    f"No data for gameweek {self.config.GAMEWEEK} "
                    f"in {csv_path}"
                )
            
            # Extract weights and convert from percentage to decimal
            form_weight = gw_data['form_dominance_smoothed'].iloc[0] / 100.0
            historic_weight = (
                gw_data['historical_weight_smoothed'].iloc[0] / 100.0
            )
            fixture_weight = (
                gw_data['fixture_sensitivity_smoothed'].iloc[0] / 100.0
            )
            
            # # Round to one decimal place
            # form_weight = round(form_weight, 1)
            # historic_weight = round(historic_weight, 1)
            # fixture_weight = round(fixture_weight, 1)
            
            # Normalise to ensure they sum to exactly 1.0
            weights = self._normalise_weights(
                form_weight, historic_weight, fixture_weight
            )
            
            if self.config.GRANULAR_OUTPUT:
                print(
                    f"✓ Loaded ML weights for {position}: "
                    f"Form={weights['form']:.1%}, "
                    f"Historic={weights['historic']:.1%}, "
                    f"Fixtures={weights['difficulty']:.1%}"
                )
            
            return weights
            
        except FileNotFoundError:
            if self.config.GRANULAR_OUTPUT:
                print(
                    f"⚠ ML weights file not found for {position}: "
                    f"{csv_path}"
                )
                print(f"  Using manual weights for {position}")
            return None
            
        except Exception as e:
            if self.config.GRANULAR_OUTPUT:
                print(
                    f"⚠ Error loading ML weights for {position}: {e}"
                )
                print(f"  Using manual weights for {position}")
            return None
    
    def _normalise_weights(self, form_weight, historic_weight, 
                          fixture_weight):
        """
        Normalise weights to ensure they sum to exactly 1.0.
        
        Args:
            form_weight (float): Form weight
            historic_weight (float): Historic weight
            fixture_weight (float): Fixture weight
            
        Returns:
            dict: Normalised weights dictionary
        """
        total = form_weight + historic_weight + fixture_weight
        
        # If total is close to 1.0, adjust the largest weight
        if abs(total - 1.0) > 0.001:  # Allow tiny rounding tolerance
            # Adjust the largest weight to make sum exactly 1.0
            weights = {
                'form': form_weight, 
                'historic': historic_weight, 
                'difficulty': fixture_weight
            }
            max_key = max(weights, key=weights.get)
            weights[max_key] = round(
                weights[max_key] + (1.0 - total), 1
            )
            
            return weights
        
        # Already close enough to 1.0
        return {
            'form': form_weight,
            'historic': historic_weight,
            'difficulty': fixture_weight
        }
    
    def load_all_weights(self, manual_weights):
        """
        Load ML weights for all positions, falling back to manual weights
        where needed.
        
        Args:
            manual_weights (dict): Manual weights dictionary with position 
                                  keys
            
        Returns:
            dict: Updated weights dictionary with ML weights where available
        """
        import copy
        
        # Start with a deep copy of manual weights
        weights = copy.deepcopy(manual_weights)
        
        # Try to load ML weights for each position
        for position in ["GK", "DEF", "MID", "FWD"]:
            ml_weights = self.load_weights_for_position(position)
            
            # If ML weights loaded successfully, use them
            if ml_weights is not None:
                weights[position] = ml_weights
        
        return weights