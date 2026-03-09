import pandas as pd

# Load the dataset containing the previously calculated Poisson features
df = pd.read_csv("turkey_poisson_features.csv")

# Create binary flags for match outcomes based on goals scored
df['h win'] = (df['FTHG'] > df['FTAG']).astype(int)
df['h draw'] = (df['FTHG'] == df['FTAG']).astype(int)
df['a win'] = (df['FTAG'] > df['FTHG']).astype(int)
df['A D'] = (df['FTAG'] == df['FTHG']).astype(int)

# Calculate points earned using the standard 3-1-0 point system
df['h p'] = (df['h win'] * 3) + (df['h draw'] * 1)
df['a p'] = (df['a win'] * 3) + (df['A D'] * 1)

# Calculate the goal difference for the match from the home team's perspective
df['gd'] = df['FTHG'] - df['FTAG']

# Aggregate total games, wins, and goals scored for teams playing at home
home_stats = df.groupby('HomeTeam').agg(
    home_games=('HomeTeam', 'count'),
    home_wins=('h win', 'sum'),
    home_goals=('FTHG', 'sum')
).reset_index()

# Aggregate total games, wins, and goals scored for teams playing away
away_stats = df.groupby('AwayTeam').agg(
    away_games=('AwayTeam', 'count'),
    away_wins=('a win', 'sum'),
    away_goals=('FTAG', 'sum')
).reset_index()

# Calculate home and away win rates as percentages
home_stats['h win rt'] = (home_stats['home_wins'] / home_stats['home_games']) * 100
away_stats['a win rt'] = (away_stats['away_wins'] / away_stats['away_games']) * 100

# Create a consolidated dataframe to calculate overall goals per game for each team
team_totals = pd.DataFrame({'team_name': home_stats['HomeTeam']})
team_totals = team_totals.merge(home_stats, left_on='team_name', right_on='HomeTeam')
team_totals = team_totals.merge(away_stats, left_on='team_name', right_on='AwayTeam')

# Calculate the overall goals per game metric across all fixtures
total_goals_scored = team_totals['home_goals'] + team_totals['away_goals']
total_matches_played = team_totals['home_games'] + team_totals['away_games']
team_totals['gpg'] = total_goals_scored / total_matches_played

# Merge the home win rates into the main dataset
df = df.merge(
    home_stats[['HomeTeam', 'h win rt']], 
    on='HomeTeam', 
    how='left'
)

# Merge the away win rates into the main dataset
df = df.merge(
    away_stats[['AwayTeam', 'a win rt']], 
    on='AwayTeam', 
    how='left'
)

# Merge the overall goals per game into the main dataset based on the home team
df = df.merge(
    team_totals[['team_name', 'gpg']], 
    left_on='HomeTeam', 
    right_on='team_name', 
    how='left'
).drop(columns=['team_name'])

# Save the fully updated dataset to a new CSV file
df.to_csv("turkey_form_and_poisson_features.csv", index=False)
print("Form features calculated and saved to turkey_form_and_poisson_features.csv")
