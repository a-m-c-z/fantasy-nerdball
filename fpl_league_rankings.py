import requests

# Get current gameweek standings
# The API endpoint for overall league standings
url = "https://fantasy.premierleague.com/api/bootstrap-static/"

# This gets general game info including current gameweek
response = requests.get(url)
data = response.json()
current_gameweek = data['events'][0]['id']  # Gets current/next gameweek

# To get specific rank information, you can use the leagues endpoint
# For the overall global league (league id is typically a very large number)
# But more practically, you can check individual pages of the standings

def get_score_for_rank(target_rank):
    """Get the score for a player at approximately the target rank"""
    # Each page has 50 entries, so calculate which page
    page = (target_rank // 50) + 1
    
    standings_url = f"https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings={page}"
    
    response = requests.get(standings_url)
    standings_data = response.json()
    
    # Find the entry closest to target rank on this page
    for entry in standings_data['standings']['results']:
        if abs(entry['rank'] - target_rank) < 25:  # Within half a page
            return {
                'rank': entry['rank'],
                'total_points': entry['total'],
                'team_name': entry['entry_name']
            }
    return None

# Get top 100k and top 10k scores
top_100k = get_score_for_rank(100000)
top_10k = get_score_for_rank(10000)

print(f"Top 100,000: Rank {top_100k['rank']} - {top_100k['total_points']} points")
print(f"Top 10,000: Rank {top_10k['rank']} - {top_10k['total_points']} points")