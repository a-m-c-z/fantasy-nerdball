#!/usr/bin/env python3
"""
Prepare ML training data from Fantasy Premier League historical data.
Processes player gameweek data with form, historical performance, and 
fixture difficulty.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import requests
from io import StringIO
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.text_utils import normalise_name
from src.data.historical_data import HistoricalDataManager
from config import Config


class MLTrainingDataPreparer:
    """Prepares training data for ML model optimisation."""
    
    def __init__(self, config, target_season, historical_seasons):
        """
        Initialise the ML training data preparer.
        
        Args:
            config: Configuration object
            target_season (str): Season to prepare data for (e.g. '2019-20')
            historical_seasons (list): Previous seasons for historical data
        """
        self.config = config
        self.target_season = target_season
        self.historical_seasons = historical_seasons
        self.base_url = (
            "https://raw.githubusercontent.com/vaastav/"
            "Fantasy-Premier-League/master/data"
        )
        self.output_dir = Path("ml/training_data") / target_season
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Columns to keep in final output
        self.keep_columns = [
            'assists', 'bps', 'clean_sheets', 'creativity', 'fixture',
            'goals_conceded', 'goals_scored', 'ict_index', 'influence',
            'minutes', 'saves', 'threat', 'total_points'
        ]
        
        # Additional columns we calculate
        self.calculated_columns = [
            'gameweek', 'form', 'historical_ppg', 'fixture_difficulty', 'opponent_team',
            'position'
        ]
        
        # Position mapping from element_type to position name
        self.position_map = {
            1: 'GK',
            2: 'DEF',
            3: 'MID',
            4: 'FWD'
        }
    
    def prepare_all_data(self):
        """
        Main method to prepare all training data.
        
        Returns:
            tuple: (success: bool, stats: dict)
        """
        print(f"\nPreparing data for {self.target_season}...")
        
        # Load team data for fixture difficulty
        teams_df = self._load_teams_data()
        if teams_df is None:
            print("Failed to load team data.")
            return False, {}
        
        # Load historical data manager
        hist_manager = HistoricalDataManager(self.config)
        
        # Get historical performance for all players
        historical_data = self._get_historical_performance(hist_manager)
        
        # Get all players for target season
        players = self._get_players_list()
        
        if players is None:
            print("Failed to fetch players list.")
            return False, {}
        
        print(f"Found {len(players)} players with 1000+ minutes")
        
        # Filter to only players WITH historical data
        players_original_count = len(players)
        players['historical_ppg'] = players['player_name'].apply(
            lambda name: self._get_historical_ppg_for_player(
                name, historical_data
            )
        )
        
        # Keep only players with historical_ppg > 0
        players_with_history = players[players['historical_ppg'] > 0].copy()
        players_without_history = players_original_count - len(
            players_with_history
        )
        
        print(f"Players with historical data: {len(players_with_history)}")
        print(f"Players without historical data (excluded): "
              f"{players_without_history}")
        
        # Process each player (only those with historical data)
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for idx, player in players_with_history.iterrows():
            try:
                player_web_name = player['player_name']
                player_dir_name = player['dir_name']
                player_id = player['id']
                player_position = player['position']
                
                # Progress indicator (less frequent for multi-season run)
                if (idx + 1) % 100 == 0:
                    print(f"  Processing player {processed_count + 1}/"
                          f"{len(players_with_history)}...")
                
                # Process player data
                success = self._process_player(
                    player_web_name, player_dir_name, player_id, 
                    player_position, teams_df, historical_data
                )
                
                if success:
                    processed_count += 1
                else:
                    skipped_count += 1
                    
            except Exception as e:
                error_count += 1
                if error_count <= 3:  # Reduced error output
                    print(f"  Error processing {player_web_name}: {e}")
        
        match_rate = (len(players_with_history) / players_original_count 
                     * 100 if players_original_count > 0 else 0)
        
        print(f"  Successfully processed: {processed_count} players")
        print(f"  Skipped (file errors): {skipped_count} players")
        print(f"  Training data includes only players with historical data")
        
        # Save a matching report
        self._save_matching_report(players, historical_data)
        
        stats = {
            'matched': len(players_with_history),
            'unmatched': players_without_history,
            'match_rate': match_rate,
            'processed': processed_count,
            'skipped': skipped_count,
            'errors': error_count
        }
        
        return True, stats
    
    def _load_teams_data(self):
        """Load teams data for fixture difficulty mapping."""
        try:
            url = f"{self.base_url}/{self.target_season}/teams.csv"
            response = requests.get(url)
            response.raise_for_status()
            teams_df = pd.read_csv(StringIO(response.text))
            
            # Create mapping of team id to strength (difficulty)
            self.team_difficulty = dict(
                zip(teams_df['id'], teams_df['strength'])
            )
            
            return teams_df
            
        except Exception as e:
            print(f"Error loading teams data: {e}")
            return None
    
    def _get_historical_performance(self, hist_manager):
        """
        Calculate historical performance for all players across seasons.
        Uses web_name from players_raw.csv for consistent naming.
        
        Args:
            hist_manager: HistoricalDataManager instance
            
        Returns:
            dict: Mapping of normalised web_names to historical PPG
        """
        historical_data = {}
        name_mapping = {}  # Track web_name -> name_key for debugging
        
        for season in self.historical_seasons:
            try:
                print(f"  Fetching data for {season}...")
                
                # Load players_raw.csv directly to get web_name
                url = f"{self.base_url}/{season}/players_raw.csv"
                response = requests.get(url)
                response.raise_for_status()
                players_raw = pd.read_csv(StringIO(response.text))
                
                # Filter to players with sufficient minutes
                players_raw = players_raw[
                    players_raw['minutes'].fillna(0) >= 500
                ].copy()
                
                for _, row in players_raw.iterrows():
                    web_name = row.get('web_name', '')
                    if not web_name:
                        continue
                    
                    # normalise web_name for consistent matching
                    name_key = normalise_name(web_name)
                    
                    # Calculate PPG
                    total_points = row.get('total_points', 0)
                    minutes = row.get('minutes', 0)
                    games_played = max(1, round(minutes / 90))
                    
                    if games_played >= 8:  # Minimum games threshold
                        ppg = total_points / games_played
                        
                        if name_key not in historical_data:
                            historical_data[name_key] = []
                            name_mapping[name_key] = web_name
                        
                        historical_data[name_key].append(ppg)
                        
            except Exception as e:
                print(f"  Warning: Could not fetch data for {season}: {e}")
        
        # Calculate weighted average PPG for each player
        weighted_historical = {}
        weights = self.config.HISTORIC_SEASON_WEIGHTS
        
        for name_key, ppg_values in historical_data.items():
            if ppg_values:
                # Use available weights for available seasons
                available_weights = weights[:len(ppg_values)]
                # Normalise weights to sum to 1.0
                weight_sum = sum(available_weights)
                normalised_weights = [
                    w / weight_sum for w in available_weights
                ]
                
                weighted_ppg = sum(
                    ppg * weight 
                    for ppg, weight in zip(ppg_values, normalised_weights)
                )
                weighted_historical[name_key] = weighted_ppg
        
        print(f"  Calculated historical data for "
              f"{len(weighted_historical)} players")
        
        # Store mapping for later use
        self.name_mapping = name_mapping
        self.historical_name_keys = set(weighted_historical.keys())
        
        return weighted_historical
    
    def _get_players_list(self):
        """Get list of all players for target season with web_name and 
        position."""
        try:
            url = f"{self.base_url}/{self.target_season}/players_raw.csv"
            response = requests.get(url)
            response.raise_for_status()
            players_df = pd.read_csv(StringIO(response.text))
            
            # Only keep players with sufficient minutes
            players_df = players_df[
                players_df['minutes'].fillna(0) >= 1000
            ].copy()
            
            # Use web_name instead of constructed name
            players_df['player_name'] = players_df['web_name']
            
            # Also keep the directory-style name for file fetching
            players_df['dir_name'] = (
                players_df['first_name'] + '_' + 
                players_df['second_name']
            )
            players_df['id'] = players_df['id']
            
            # Map element_type to position name
            players_df['position'] = players_df['element_type'].map(
                self.position_map
            )
            
            return players_df[['player_name', 'dir_name', 'id', 'position']]
            
        except Exception as e:
            print(f"Error fetching players list: {e}")
            return None
    
    def _process_player(self, player_web_name, player_dir_name, player_id, 
                       player_position, teams_df, historical_data):
        """
        Process individual player's gameweek data.
        
        Args:
            player_web_name (str): Player's web_name (for matching)
            player_dir_name (str): Player's directory name (for fetching)
            player_id (int): Player's ID
            player_position (str): Player's position (GK/DEF/MID/FWD)
            teams_df (pd.DataFrame): Teams data for fixture difficulty
            historical_data (dict): Historical PPG by normalised web_name
            
        Returns:
            bool: True if processed successfully, False if skipped
        """
        try:
            # Load player's gameweek data using directory name
            url = (
                f"{self.base_url}/{self.target_season}/players/"
                f"{player_dir_name}_{player_id}/gw.csv"
            )
            response = requests.get(url)
            response.raise_for_status()
            gw_df = pd.read_csv(StringIO(response.text))
            
            if gw_df.empty:
                return False
            
            # Calculate form (rolling 5-game average of total_points)
            gw_df['form'] = gw_df['total_points'].rolling(
                window=5, min_periods=1
            ).mean().round(1)
            
            # Add fixture difficulty
            gw_df['fixture_difficulty'] = gw_df['opponent_team'].map(
                self.team_difficulty
            ).fillna(3)  # Default to 3 if not found

            # Add gameweek
            gw_df['gameweek'] = range(1, len(gw_df) + 1)
            
            # Add historical PPG using web_name for matching
            # Note: At this point we know hist_ppg > 0 from earlier check
            historical_ppg = self._get_historical_ppg_for_player(
                player_web_name, historical_data
            )
            
            gw_df['historical_ppg'] = historical_ppg
            
            # Add position (constant for all rows of this player)
            gw_df['position'] = player_position
            
            # Keep only required columns
            columns_to_keep = (
                self.keep_columns + 
                ['form', 'historical_ppg', 'fixture_difficulty', 'position', 'gameweek']
            )
            
            # Only keep columns that exist
            existing_columns = [
                col for col in columns_to_keep if col in gw_df.columns
            ]
            gw_df = gw_df[existing_columns]
            
            # Drop rows with missing critical values
            gw_df = gw_df.dropna(subset=['total_points', 'minutes'])
            
            if gw_df.empty:
                return False
            
            # Save to output directory using web_name for filename
            sanitised_name = self._sanitise_filename(player_web_name)
            output_path = self.output_dir / f"{sanitised_name}.csv"
            gw_df.to_csv(output_path, index=False)
            
            return True
            
        except Exception as e:
            # Most common error will be 404 for players not in this season
            if "404" not in str(e):
                raise  # Re-raise unexpected errors
            return False
    
    def _get_historical_ppg_for_player(self, player_web_name, 
                                      historical_data):
        """
        Get historical PPG for a player using web_name.
        
        Args:
            player_web_name (str): Player's web_name
            historical_data (dict): Historical PPG by normalised web_name
            
        Returns:
            float: Historical PPG or 0.0 if not found
        """
        # normalise the web_name
        name_key = normalise_name(player_web_name)
        
        # Direct lookup (should work for most players now)
        return historical_data.get(name_key, 0.0)
    
    def _sanitise_filename(self, name):
        """
        Sanitise player name for use as filename.
        
        Args:
            name (str): Player name
            
        Returns:
            str: Sanitised filename
        """
        # Replace underscores with spaces first
        name = name.replace('_', ' ')
        # Normalise the name
        name = normalise_name(name)
        # Replace spaces with underscores for filename
        name = name.replace(' ', '_')
        return name
    
    def _save_matching_report(self, players, historical_data):
        """
        Save a report showing which players matched historical data.
        
        Args:
            players (pd.DataFrame): Current season players
            historical_data (dict): Historical PPG data
        """
        report_path = self.output_dir.parent / "matching_report.csv"
        
        report_data = []
        for _, player in players.iterrows():
            player_web_name = player['player_name']
            player_dir_name = player['dir_name']
            hist_ppg = self._get_historical_ppg_for_player(
                player_web_name, historical_data
            )
            
            # Get normalised name
            name_norm = normalise_name(player_web_name)
            
            report_data.append({
                'web_name': player_web_name,
                'directory_name': player_dir_name,
                'normalised_name': name_norm,
                'historical_ppg': hist_ppg,
                'matched': 'Yes' if hist_ppg > 0 else 'No'
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(report_path, index=False)
        print(f"\n📊 Matching report saved to: {report_path}")


def main():
    """Main function to prepare ML training data for multiple seasons."""
    config = Config()
    
    # Define all seasons to process with their historical seasons
    seasons_to_process = [
        ("2019-20", ["2016-17", "2017-18", "2018-19"]),
        ("2020-21", ["2017-18", "2018-19", "2019-20"]),
        ("2021-22", ["2018-19", "2019-20", "2020-21"]),
        ("2022-23", ["2019-20", "2020-21", "2021-22"]),
        ("2023-24", ["2020-21", "2021-22", "2022-23"]),
        ("2024-25", ["2021-22", "2022-23", "2023-24"]),
    ]
    
    print("="*60)
    print("ML TRAINING DATA PREPARATION - MULTI-SEASON")
    print("="*60)
    print(f"\nProcessing {len(seasons_to_process)} seasons")
    print("Minimum minutes per season: 1000")
    print()
    
    overall_stats = {
        'seasons_processed': 0,
        'seasons_failed': 0,
        'total_players': 0,
        'total_matched': 0,
        'total_unmatched': 0
    }
    
    for target_season, historical_seasons in seasons_to_process:
        print(f"\n{'='*60}")
        print(f"SEASON: {target_season}")
        print(f"Historical data from: {', '.join(historical_seasons)}")
        print(f"{'='*60}")
        
        try:
            preparer = MLTrainingDataPreparer(
                config, target_season, historical_seasons
            )
            
            # Process this season's data
            success, stats = preparer.prepare_all_data()
            
            if success:
                overall_stats['seasons_processed'] += 1
                overall_stats['total_players'] += (
                    stats['matched'] + stats['unmatched']
                )
                overall_stats['total_matched'] += stats['matched']
                overall_stats['total_unmatched'] += stats['unmatched']
                
                print(f"\n✓ {target_season} complete - "
                      f"{stats['matched']}/{stats['matched']+stats['unmatched']} "
                      f"players matched ({stats['match_rate']:.1f}%)")
            else:
                overall_stats['seasons_failed'] += 1
                print(f"\n✗ {target_season} failed")
                
        except Exception as e:
            overall_stats['seasons_failed'] += 1
            print(f"\n✗ Error processing {target_season}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Seasons successfully processed: "
          f"{overall_stats['seasons_processed']}")
    print(f"Seasons failed: {overall_stats['seasons_failed']}")
    print(f"Total players evaluated: "
          f"{overall_stats['total_players']}")
    print(f"Players included in training data: "
          f"{overall_stats['total_matched']}")
    print(f"Players excluded (no historical data): "
          f"{overall_stats['total_unmatched']}")
    
    if overall_stats['total_players'] > 0:
        overall_match_rate = (
            overall_stats['total_matched'] / 
            overall_stats['total_players'] * 100
        )
        print(f"Overall inclusion rate: {overall_match_rate:.1f}%")
    
    print(f"\n Only players with historical PPG data are included")
    print(f"   This ensures the model can learn from historical patterns")
    
    print("\n Multi-season data preparation complete!")
    print(f"Training data saved to: ml/training_data/{{season}}/")


if __name__ == "__main__":
    main()