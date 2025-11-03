"""Script to fetch FPL global league rankings."""

import requests
import sys


def get_current_gameweek():
    """Get the current gameweek number."""
    try:
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Find current gameweek (is_current=True)
        for event in data['events']:
            if event['is_current']:
                return event['id']
        
        # If no current, find next
        for event in data['events']:
            if event['is_next']:
                return event['id']
        
        return None
    except Exception as e:
        print(f"Error getting current gameweek: {e}")
        return None


def get_total_players():
    """
    Get the total number of players in the global league.
    
    Returns:
        int or None: Total number of players
    """
    try:
        # Get bootstrap-static data which contains total_players
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        
        response = requests.get(url, params={}, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Get total players from the API
        total = data.get('total_players', None)
        
        if total:
            return total
        
        return None
        
    except Exception as e:
        print(f"Error fetching total players: {e}")
        return None


def get_score_for_rank(target_rank):
    """
    Get the score for a player at approximately the target rank.
    
    Args:
        target_rank (int): Target rank to find
        
    Returns:
        dict or None: Player info at that rank, or None if not found
    """
    try:
        # Each page has 50 entries
        # Page 1: ranks 1-50, Page 2: ranks 51-100, etc.
        # So for rank 10000: (10000-1)//50 + 1 = 199 + 1 = 200
        page = ((target_rank - 1) // 50) + 1
        
        # Global league ID is 314
        url = (
            "https://fantasy.premierleague.com/api/"
            "leagues-classic/314/standings/"
        )
        params = {'page_standings': page}
        
        print(f"Fetching page {page} for rank ~{target_rank:,}...")
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        standings_data = response.json()
        
        # Check if we have standings
        standings = standings_data.get('standings', {}).get('results', [])
        
        if not standings:
            print(f"  ⚠️  No standings data found on page {page}")
            return None
        
        # Find the entry closest to target rank
        closest_entry = None
        min_diff = float('inf')
        
        for entry in standings:
            rank_diff = abs(entry['rank'] - target_rank)
            if rank_diff < min_diff:
                min_diff = rank_diff
                closest_entry = entry
        
        if closest_entry:
            return {
                'rank': closest_entry['rank'],
                'total_points': closest_entry['total'],
                'team_name': closest_entry['entry_name'],
                'player_name': closest_entry['player_name']
            }
        else:
            print(f"  ⚠️  Could not find any entry on page {page}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"  ⚠️  Request timed out for rank {target_rank:,}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  API error for rank {target_rank:,}: {e}")
        return None
    except Exception as e:
        print(f"  ⚠️  Unexpected error for rank {target_rank:,}: {e}")
        return None


def main():
    """Main function to fetch and display rankings."""
    print("=" * 60)
    print("FPL Global League Rankings")
    print("=" * 60)
    
    # Get current gameweek
    current_gw = get_current_gameweek()
    if current_gw:
        print(f"\nCurrent Gameweek: {current_gw}")
    else:
        print("\n⚠️  Could not determine current gameweek")
    
    # Get total players
    total_players = get_total_players()
    if total_players:
        print(f"Total Players: {total_players:,}")
    else:
        print("⚠️  Could not determine total players")
    
    print("\nFetching rankings...\n")
    
    # Get top 10k score
    top_10k = get_score_for_rank(10000)
    
    # Get top 100k score
    top_100k = get_score_for_rank(100000)
    
    # Get percentile scores if we have total players
    percentiles = {}
    if total_players:
        print("\nFetching percentile rankings...\n")
        
        p01_rank = int(total_players * 0.01)
        p10_rank = int(total_players * 0.10)
        p25_rank = int(total_players * 0.25)
        p50_rank = int(total_players * 0.50)
        p75_rank = int(total_players * 0.75)
        
        percentiles['p01'] = get_score_for_rank(p01_rank)
        percentiles['p10'] = get_score_for_rank(p10_rank)
        percentiles['p25'] = get_score_for_rank(p25_rank)
        percentiles['p50'] = get_score_for_rank(p50_rank)
        percentiles['p75'] = get_score_for_rank(p75_rank)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if top_10k:
        print(f"\n✅ Top 10,000:")
        print(f"   Rank: {top_10k['rank']:,}")
        print(f"   Points: {top_10k['total_points']}")
        print(f"   Team: {top_10k['team_name']}")
        print(f"   Manager: {top_10k['player_name']}")
    else:
        print(f"\n❌ Top 10,000: Could not fetch data")
    
    if top_100k:
        print(f"\n✅ Top 100,000:")
        print(f"   Rank: {top_100k['rank']:,}")
        print(f"   Points: {top_100k['total_points']}")
        print(f"   Team: {top_100k['team_name']}")
        print(f"   Manager: {top_100k['player_name']}")
    else:
        print(f"\n❌ Top 100,000: Could not fetch data")
    
    # Display percentile results
    if percentiles:
        percentile_labels = {
            'p01': '1st Percentile',
            'p10': '10th Percentile',
            'p25': '25th Percentile',
            'p50': '50th Percentile (Median)',
            'p75': '75th Percentile'
        }
        
        for key, label in percentile_labels.items():
            if percentiles.get(key):
                data = percentiles[key]
                print(f"\n✅ {label}:")
                print(f"   Rank: {data['rank']:,}")
                print(f"   Points: {data['total_points']}")
                print(f"   Team: {data['team_name']}")
                print(f"   Manager: {data['player_name']}")
            else:
                print(f"\n❌ {label}: Could not fetch data")
    
    print("\n" + "=" * 60)
    
    # Return codes for scripting
    if top_10k and top_100k:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())