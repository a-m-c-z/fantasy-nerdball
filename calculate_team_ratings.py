#!/usr/bin/env python3
"""
Calculate sophisticated attacking and defensive ratings for teams based on 
actual performance relative to fixture difficulty.

This script analyzes fixtures.csv to determine:
1. Attacking Rating: How well a team scores relative to opponent strength
2. Defensive Rating: How well a team defends relative to opponent strength

Higher attacking rating = team scores more goals than expected
Higher defensive rating = team concedes fewer goals than expected
"""

import pandas as pd
import numpy as np
from config import Config


class TeamRatingsCalculator:
    """Calculate team performance ratings from fixture data."""
    
    def __init__(self, config):
        self.config = config
        self.fixtures_df = None
        self.team_ratings = {}
        
    def load_fixtures(self):
        """Load fixtures data from CSV."""
        try:
            self.fixtures_df = pd.read_csv('data/fixtures.csv')
            print(f"Loaded {len(self.fixtures_df)} fixtures from data/fixtures.csv")
            return True
        except FileNotFoundError:
            print("Error: data/fixtures.csv not found. Please run main.py first to generate fixture data.")
            return False
        except Exception as e:
            print(f"Error loading fixtures: {e}")
            return False
    
    def filter_completed_fixtures(self):
        """Filter to only completed fixtures before current gameweek."""
        if self.fixtures_df is None:
            return
        
        # Filter to completed fixtures before current gameweek
        completed = self.fixtures_df[
            (self.fixtures_df['finished'] == True) &
            (self.fixtures_df['gameweek'] < self.config.GAMEWEEK)
        ].copy()
        
        print(f"\nAnalyzing {len(completed)} completed fixtures before GW{self.config.GAMEWEEK}")
        
        if len(completed) == 0:
            print("Warning: No completed fixtures found. Ratings will be based on neutral values.")
        
        self.fixtures_df = completed
        
    def calculate_expected_performance(self, difficulty, is_home=True):
        """
        Calculate expected goals based on fixture difficulty.
        
        FPL difficulty scale: 1 (easiest) to 5 (hardest)
        
        Args:
            difficulty (int): FPL fixture difficulty (1-5)
            is_home (bool): Whether playing at home
            
        Returns:
            float: Expected goals
        """
        # Base expected goals by difficulty
        # Difficulty 1 (easiest) = expect ~2.5 goals
        # Difficulty 5 (hardest) = expect ~0.8 goals
        base_expected = {
            1: 2.5,
            2: 2.0,
            3: 1.5,
            4: 1.2,
            5: 0.8
        }
        
        expected = base_expected.get(difficulty, 1.5)
        
        # Home advantage: ~0.3 goals
        if is_home:
            expected += 0.3
        
        return expected
    
    def calculate_team_attacking_rating(self, team_name):
        """
        Calculate attacking rating for a team.
        
        Rating > 1.0 = scoring more than expected
        Rating < 1.0 = scoring less than expected
        Rating = 1.0 = neutral/average
        
        Args:
            team_name (str): Team name
            
        Returns:
            float: Attacking rating
        """
        if self.fixtures_df is None or len(self.fixtures_df) == 0:
            return 1.0
        
        # Get all fixtures for this team
        home_fixtures = self.fixtures_df[
            self.fixtures_df['home_team'] == team_name
        ].copy()
        away_fixtures = self.fixtures_df[
            self.fixtures_df['away_team'] == team_name
        ].copy()
        
        total_actual_goals = 0
        total_expected_goals = 0
        fixture_count = 0
        
        # Process home fixtures
        for _, fixture in home_fixtures.iterrows():
            if pd.notna(fixture['home_score']):
                actual_goals = fixture['home_score']
                expected_goals = self.calculate_expected_performance(
                    fixture['home_difficulty'], is_home=True
                )
                
                total_actual_goals += actual_goals
                total_expected_goals += expected_goals
                fixture_count += 1
        
        # Process away fixtures
        for _, fixture in away_fixtures.iterrows():
            if pd.notna(fixture['away_score']):
                actual_goals = fixture['away_score']
                expected_goals = self.calculate_expected_performance(
                    fixture['away_difficulty'], is_home=False
                )
                
                total_actual_goals += actual_goals
                total_expected_goals += expected_goals
                fixture_count += 1
        
        if fixture_count == 0 or total_expected_goals == 0:
            return 1.0
        
        # Calculate ratio of actual to expected
        attacking_rating = total_actual_goals / total_expected_goals
        
        # Apply reasonable bounds (0.5 to 2.0)
        attacking_rating = max(0.5, min(2.0, attacking_rating))
        
        return attacking_rating
    
    def calculate_team_defensive_rating(self, team_name):
        """
        Calculate defensive rating for a team.
        
        Rating > 1.0 = conceding fewer goals than expected (good defence)
        Rating < 1.0 = conceding more goals than expected (poor defence)
        Rating = 1.0 = neutral/average
        
        Args:
            team_name (str): Team name
            
        Returns:
            float: Defensive rating
        """
        if self.fixtures_df is None or len(self.fixtures_df) == 0:
            return 1.0
        
        # Get all fixtures for this team
        home_fixtures = self.fixtures_df[
            self.fixtures_df['home_team'] == team_name
        ].copy()
        away_fixtures = self.fixtures_df[
            self.fixtures_df['away_team'] == team_name
        ].copy()
        
        total_actual_conceded = 0
        total_expected_conceded = 0
        fixture_count = 0
        
        # Process home fixtures (conceded = away team's score)
        for _, fixture in home_fixtures.iterrows():
            if pd.notna(fixture['away_score']):
                actual_conceded = fixture['away_score']
                # Expected goals conceded based on opponent's difficulty
                expected_conceded = self.calculate_expected_performance(
                    fixture['away_difficulty'], is_home=False
                )
                
                total_actual_conceded += actual_conceded
                total_expected_conceded += expected_conceded
                fixture_count += 1
        
        # Process away fixtures (conceded = home team's score)
        for _, fixture in away_fixtures.iterrows():
            if pd.notna(fixture['home_score']):
                actual_conceded = fixture['home_score']
                # Expected goals conceded based on opponent's difficulty
                expected_conceded = self.calculate_expected_performance(
                    fixture['home_difficulty'], is_home=True
                )
                
                total_actual_conceded += actual_conceded
                total_expected_conceded += expected_conceded
                fixture_count += 1
        
        if fixture_count == 0 or total_expected_conceded == 0:
            return 1.0
        
        # Calculate ratio: higher is better (conceding less than expected)
        # Invert the ratio so > 1.0 = good defence
        defensive_rating = total_expected_conceded / total_actual_conceded
        
        # Handle edge case of zero goals conceded
        if total_actual_conceded == 0:
            defensive_rating = 2.0  # Maximum rating for perfect defence
        
        # Apply reasonable bounds (0.5 to 2.0)
        defensive_rating = max(0.5, min(2.0, defensive_rating))
        
        return defensive_rating
    
    def calculate_all_team_ratings(self):
        """Calculate ratings for all teams in the dataset."""
        if self.fixtures_df is None or len(self.fixtures_df) == 0:
            print("\nNo fixture data available. Using neutral ratings (1.0) for all teams.")
            # Get teams from config
            for team in self.config.TEAM_MODIFIERS.keys():
                self.team_ratings[team] = {
                    'attacking_rating': 1.0,
                    'defensive_rating': 1.0,
                    'fixtures_played': 0,
                    'goals_scored': 0,
                    'goals_conceded': 0
                }
            return
        
        # Get unique teams
        home_teams = set(self.fixtures_df['home_team'].unique())
        away_teams = set(self.fixtures_df['away_team'].unique())
        all_teams = home_teams.union(away_teams)
        
        print(f"\nCalculating ratings for {len(all_teams)} teams...")
        
        for team in sorted(all_teams):
            attacking_rating = self.calculate_team_attacking_rating(team)
            defensive_rating = self.calculate_team_defensive_rating(team)
            
            # Get actual stats for context
            team_home = self.fixtures_df[self.fixtures_df['home_team'] == team]
            team_away = self.fixtures_df[self.fixtures_df['away_team'] == team]
            
            goals_scored = (
                team_home['home_score'].sum() + 
                team_away['away_score'].sum()
            )
            goals_conceded = (
                team_home['away_score'].sum() + 
                team_away['home_score'].sum()
            )
            fixtures_played = len(team_home) + len(team_away)
            
            self.team_ratings[team] = {
                'attacking_rating': round(attacking_rating, 2),
                'defensive_rating': round(defensive_rating, 2),
                'fixtures_played': fixtures_played,
                'goals_scored': int(goals_scored),
                'goals_conceded': int(goals_conceded),
                'goals_per_game': round(goals_scored / max(fixtures_played, 1), 2),
                'conceded_per_game': round(goals_conceded / max(fixtures_played, 1), 2)
            }
    
    def print_ratings_summary(self):
        """Print formatted summary of team ratings."""
        if not self.team_ratings:
            print("\nNo ratings calculated.")
            return
        
        print("\n" + "=" * 100)
        print(f"TEAM PERFORMANCE RATINGS (Before GW{self.config.GAMEWEEK})")
        print("=" * 100)
        
        print("\nRating Guide:")
        print("  Attacking Rating > 1.0 = Scoring more goals than expected (good for FWD/MID)")
        print("  Attacking Rating < 1.0 = Scoring fewer goals than expected (bad for FWD/MID)")
        print("  Defensive Rating > 1.0 = Conceding fewer goals than expected (good for GK/DEF)")
        print("  Defensive Rating < 1.0 = Conceding more goals than expected (bad for GK/DEF)")
        
        # Convert to DataFrame for better display
        ratings_df = pd.DataFrame.from_dict(self.team_ratings, orient='index')
        ratings_df = ratings_df.sort_values('attacking_rating', ascending=False)
        
        print("\n" + "-" * 100)
        print(f"{'Team':<20} {'Attack':<10} {'Defence':<10} {'GP':<6} {'GF':<6} {'GA':<6} {'GF/G':<8} {'GA/G':<8}")
        print("-" * 100)
        
        for team, data in ratings_df.iterrows():
            # Text indicators for ratings
            attack_symbol = "***" if data['attacking_rating'] > 1.2 else "**" if data['attacking_rating'] > 1.0 else "*" if data['attacking_rating'] < 0.8 else "  "
            defence_symbol = "***" if data['defensive_rating'] > 1.2 else "**" if data['defensive_rating'] > 1.0 else "*" if data['defensive_rating'] < 0.8 else "  "
            
            print(f"{team:<20} {data['attacking_rating']:<4.2f} {attack_symbol:<5} {data['defensive_rating']:<4.2f} {defence_symbol:<5} "
                  f"{data['fixtures_played']:<6} {data['goals_scored']:<6} {data['goals_conceded']:<6} "
                  f"{data['goals_per_game']:<8.2f} {data['conceded_per_game']:<8.2f}")
        
        print("-" * 100)
        print("\nRating Indicators:")
        print("  *** = Very strong (>1.2)")
        print("  **  = Good (>1.0)")
        print("  *   = Poor (<0.8)")
        
        # Top/Bottom 5 summaries
        print("\nTOP 5 ATTACKING TEAMS (Best for FWD/MID assets):")
        top_attack = ratings_df.nlargest(5, 'attacking_rating')
        for idx, (team, data) in enumerate(top_attack.iterrows(), 1):
            print(f"  {idx}. {team:<20} Rating: {data['attacking_rating']:.2f} ({data['goals_per_game']:.2f} goals/game)")
        
        print("\nTOP 5 DEFENSIVE TEAMS (Best for GK/DEF assets):")
        top_defence = ratings_df.nlargest(5, 'defensive_rating')
        for idx, (team, data) in enumerate(top_defence.iterrows(), 1):
            print(f"  {idx}. {team:<20} Rating: {data['defensive_rating']:.2f} ({data['conceded_per_game']:.2f} conceded/game)")
        
        print("\nBOTTOM 5 ATTACKING TEAMS (Avoid FWD/MID assets):")
        bottom_attack = ratings_df.nsmallest(5, 'attacking_rating')
        for idx, (team, data) in enumerate(bottom_attack.iterrows(), 1):
            print(f"  {idx}. {team:<20} Rating: {data['attacking_rating']:.2f} ({data['goals_per_game']:.2f} goals/game)")
        
        print("\nBOTTOM 5 DEFENSIVE TEAMS (Avoid GK/DEF assets, target with FWD/MID):")
        bottom_defence = ratings_df.nsmallest(5, 'defensive_rating')
        for idx, (team, data) in enumerate(bottom_defence.iterrows(), 1):
            print(f"  {idx}. {team:<20} Rating: {data['defensive_rating']:.2f} ({data['conceded_per_game']:.2f} conceded/game)")
        
        print("\n" + "=" * 100)
    
    def save_ratings_to_csv(self, filename='data/team_ratings.csv'):
        """Save ratings to CSV file for integration with main model."""
        if not self.team_ratings:
            print("\nNo ratings to save.")
            return
        
        ratings_df = pd.DataFrame.from_dict(self.team_ratings, orient='index')
        ratings_df.index.name = 'team'
        ratings_df = ratings_df.sort_values('attacking_rating', ascending=False)
        
        try:
            ratings_df.to_csv(filename)
            print(f"\nRatings saved to {filename}")
            print(f"   This file can be used to integrate sophisticated fixture ratings into the main model.")
        except Exception as e:
            print(f"\nError saving ratings: {e}")
    
    def run(self):
        """Main execution flow."""
        print("=" * 100)
        print("TEAM RATINGS CALCULATOR")
        print("=" * 100)
        print(f"\nAnalyzing team performance up to GW{self.config.GAMEWEEK}")
        print("This will calculate attacking and defensive ratings based on actual performance")
        print("relative to fixture difficulty.")
        
        # Load fixtures
        if not self.load_fixtures():
            return False
        
        # Filter to completed fixtures
        self.filter_completed_fixtures()
        
        # Calculate ratings
        self.calculate_all_team_ratings()
        
        # Display results
        self.print_ratings_summary()
        
        # Save to CSV
        self.save_ratings_to_csv()
        
        return True


def main():
    """Main entry point."""
    try:
        config = Config()
        calculator = TeamRatingsCalculator(config)
        success = calculator.run()
        
        if success:
            print("\nTeam ratings calculated successfully!")
            print("\nNext steps:")
            print("  1. Review the ratings above")
            print("  2. Check data/team_ratings.csv for the saved data")
            print("  3. Integrate these ratings into the fixture difficulty calculation")
        else:
            print("\nFailed to calculate team ratings.")
            print("   Make sure to run main.py first to generate data/fixtures.csv")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()