"""Module for tracking FPL global standings cutoffs."""

import os
import pandas as pd
import requests
from datetime import datetime


class StandingsTracker:
    """Handles fetching and tracking of FPL global standings cutoffs."""
    
    # Target ranks to track
    TARGET_RANKS = {
        'top_10k': 10000,
        'top_100k': 100000
    }
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://fantasy.premierleague.com/api"
        self.data_file = "data/standings_history.csv"
        self.total_players = None
        self.percentile_ranks = {}
        self._ensure_data_file()
    
    def _ensure_data_file(self):
        """Create standings history file if it doesn't exist."""
        if not os.path.exists(self.data_file):
            os.makedirs("data", exist_ok=True)
            df = pd.DataFrame(columns=[
                'gameweek', 'top_10k', 'top_100k', 'p25', 'p50', 'p75',
                'timestamp', 'data_quality'
            ])
            df.to_csv(self.data_file, index=False)
    
    def get_total_players(self):
        """
        Get the total number of players in the global league.
        
        Returns:
            int or None: Total number of players, or None if unavailable
        """
        try:
            # Get bootstrap-static data which contains total_players
            url = f"{self.base_url}/bootstrap-static/"
            
            response = requests.get(url, params={}, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Get total players from the API
            total = data.get('total_players', None)
            
            if total:
                return total
            
            return None
            
        except Exception as e:
            if self.config.GRANULAR_OUTPUT:
                print(f"Error fetching total players: {e}")
            return None
    
    def calculate_percentile_ranks(self):
        """
        Calculate rank values for 25th, 50th, and 75th percentiles.
        Updates self.percentile_ranks.
        """
        if self.total_players is None:
            self.total_players = self.get_total_players()
        
        if self.total_players:
            self.percentile_ranks = {
                'p25': int(self.total_players * 0.25),
                'p50': int(self.total_players * 0.50),
                'p75': int(self.total_players * 0.75)
            }
            
            if self.config.GRANULAR_OUTPUT:
                print(f"\nTotal players: {self.total_players:,}")
                print(f"Percentile ranks: "
                      f"25th={self.percentile_ranks['p25']:,}, "
                      f"50th={self.percentile_ranks['p50']:,}, "
                      f"75th={self.percentile_ranks['p75']:,}")
    
    def get_rank_cutoff_score(self, target_rank):
        """
        Get the total points for a player at approximately the target 
        rank.
        
        Args:
            target_rank (int): The target rank (e.g., 10000, 50000, 
                              100000)
            
        Returns:
            int or None: Total points at that rank, or None if 
                        unavailable
        """
        try:
            # Calculate which page the rank would be on (50 entries per 
            # page)
            page = (target_rank // 50) + 1
            
            # Global league ID is 314
            url = f"{self.base_url}/leagues-classic/314/standings/"
            params = {'page_standings': page}
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Find the entry closest to our target rank
            standings = data.get('standings', {}).get('results', [])
            
            if not standings:
                return None
            
            # Find entry closest to target rank
            closest_entry = min(
                standings, 
                key=lambda x: abs(x.get('rank', 0) - target_rank)
            )
            
            return closest_entry.get('total', None)
            
        except Exception as e:
            if self.config.GRANULAR_OUTPUT:
                print(f"Error fetching rank {target_rank} cutoff: {e}")
            return None
    
    def fetch_current_cutoffs(self):
        """
        Fetch current gameweek cutoff scores for all target ranks and 
        percentiles.
        
        Returns:
            dict: Dictionary with rank cutoffs {'top_10k': score, ...}
        """
        cutoffs = {}
        
        if self.config.GRANULAR_OUTPUT:
            print(f"\nFetching global standings cutoffs...")
        
        # Fetch fixed rank cutoffs
        for rank_name, rank_value in self.TARGET_RANKS.items():
            score = self.get_rank_cutoff_score(rank_value)
            cutoffs[rank_name] = score
            
            if self.config.GRANULAR_OUTPUT and score is not None:
                display_name = rank_name.replace('_', ' ').title()
                print(f"  {display_name}: {score} pts")
        
        # Calculate and fetch percentile cutoffs
        self.calculate_percentile_ranks()
        
        if self.percentile_ranks:
            for percentile_name, rank_value in (
                self.percentile_ranks.items()
            ):
                score = self.get_rank_cutoff_score(rank_value)
                cutoffs[percentile_name] = score
                
                if self.config.GRANULAR_OUTPUT and score is not None:
                    percentile_num = percentile_name.replace('p', '')
                    percentile_label = f"{percentile_num}th percentile"
                    print(f"  {percentile_label}: {score} pts "
                          f"(rank ~{rank_value:,})")
        
        return cutoffs
    
    def save_cutoffs_for_gameweek(self, gameweek, cutoffs):
        """
        Save cutoff scores for a specific gameweek.
        
        Args:
            gameweek (int): Gameweek number
            cutoffs (dict): Dictionary of cutoff scores
        """
        try:
            # Load existing data
            df = pd.read_csv(self.data_file)
            
            # Remove any existing entry for this gameweek
            df = df[df['gameweek'] != gameweek]
            
            # Create new entry
            new_entry = {
                'gameweek': gameweek,
                'top_10k': cutoffs.get('top_10k'),
                'top_100k': cutoffs.get('top_100k'),
                'p25': cutoffs.get('p25'),
                'p50': cutoffs.get('p50'),
                'p75': cutoffs.get('p75'),
                'timestamp': datetime.now().isoformat(),
                'data_quality': 'actual'
            }
            
            # Append and save
            df = pd.concat(
                [df, pd.DataFrame([new_entry])], 
                ignore_index=True
            )
            df = df.sort_values('gameweek')
            df.to_csv(self.data_file, index=False)
            
            if self.config.GRANULAR_OUTPUT:
                print(f"✅ Standings cutoffs saved for GW{gameweek}")
                
        except Exception as e:
            if self.config.GRANULAR_OUTPUT:
                print(f"Error saving cutoffs: {e}")
    
    def get_historical_cutoffs(self):
        """
        Load historical cutoff data.
        
        Returns:
            pd.DataFrame: Historical cutoffs with gameweek, scores, and 
                         quality
        """
        try:
            df = pd.read_csv(self.data_file)
            return df
        except Exception:
            return pd.DataFrame(columns=[
                'gameweek', 'top_10k', 'top_100k', 'p25', 'p50', 'p75',
                'timestamp', 'data_quality'
            ])
    
    def interpolate_missing_data(self, current_gameweek):
        """
        Interpolate missing historical data using linear interpolation.
        
        Args:
            current_gameweek (int): Current gameweek number
            
        Returns:
            pd.DataFrame: Complete data with interpolated values
        """
        df = self.get_historical_cutoffs()
        
        if df.empty:
            return df
        
        # Create complete gameweek range
        all_gameweeks = list(range(1, current_gameweek))
        complete_df = pd.DataFrame({'gameweek': all_gameweeks})
        
        # Merge with existing data
        df = complete_df.merge(df, on='gameweek', how='left')
        
        # Mark existing data quality
        df.loc[df['data_quality'].notna(), 'data_quality'] = 'actual'
        
        # Interpolate missing values for each rank category
        rank_cols = ['top_10k', 'top_100k', 'p25', 'p50', 'p75']
        for rank_col in rank_cols:
            if rank_col in df.columns:
                # Linear interpolation
                df[rank_col] = df[rank_col].interpolate(
                    method='linear', 
                    limit_direction='both'
                )
                
                # Mark interpolated values
                mask = df['data_quality'].isna()
                df.loc[mask, 'data_quality'] = 'interpolated'
        
        # Fill any remaining NaN in data_quality
        df['data_quality'] = df['data_quality'].fillna('interpolated')
        
        return df
    
    def get_complete_history_for_plotting(self, current_gameweek):
        """
        Get complete historical data ready for plotting.
        Converts to cumulative points and includes gameweek 0.
        
        Args:
            current_gameweek (int): Current gameweek number
            
        Returns:
            pd.DataFrame: Complete data with cumulative points
        """
        df = self.interpolate_missing_data(current_gameweek)
        
        if df.empty:
            return df
        
        # Convert to cumulative points
        for col in ['top_10k', 'top_100k', 'p25', 'p50', 'p75']:
            if col in df.columns:
                # These are already cumulative from the API
                pass
        
        # Add gameweek 0 with 0 points for all ranks
        gw0_row = pd.DataFrame([{
            'gameweek': 0,
            'top_10k': 0,
            'top_100k': 0,
            'p25': 0,
            'p50': 0,
            'p75': 0,
            'data_quality': 'actual'
        }])
        
        df = pd.concat([gw0_row, df], ignore_index=True)
        df = df.sort_values('gameweek').reset_index(drop=True)
        
        if self.config.GRANULAR_OUTPUT and not df.empty:
            actual_count = (
                len(df[df['data_quality'] == 'actual']) - 1
            )  # Exclude GW0
            interpolated_count = (
                len(df[df['data_quality'] == 'interpolated'])
            )
            print(f"\nStandings data: {actual_count} actual, "
                  f"{interpolated_count} interpolated (+ GW0 baseline)")
        
        return df