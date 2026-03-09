import pandas as pd

df = pd.read_csv("turkey_overall_probabilities.csv")


# A standard starting rho value in football modeling is typically around -0.15
rho = -0.15

# Store original probabilities to accurately adjust the overall match outcomes
old_p00 = df['P(0 0)'].copy()
old_p10 = df['P(1 0)'].copy()
old_p01 = df['P(0 1)'].copy()
old_p11 = df['P(1 1)'].copy()

# Apply the Dixon-Coles tau adjustment functions to the low-scoring probabilities
df['P(0 0)'] = old_p00 * (1 - df['lambda h'] * df['lambda a'] * rho)
df['P(1 0)'] = old_p10 * (1 + df['lambda a'] * rho)
df['P(0 1)'] = old_p01 * (1 + df['lambda h'] * rho)
df['P(1 1)'] = old_p11 * (1 - rho)


df['P(d)'] = df['P(d)'] - old_p00 - old_p11 + df['P(0 0)'] + df['P(1 1)']
df['P(h)'] = df['P(h)'] - old_p10 + df['P(1 0)']
df['P(A)'] = df['P(A)'] - old_p01 + df['P(0 1)']

# CORNERS AND YELLOW CARDS EXPECTED VALUES
# Calculate global league averages for secondary metrics
league_hc_avg = df['HC'].mean()
league_ac_avg = df['AC'].mean()
league_hy_avg = df['HY'].mean()
league_ay_avg = df['AY'].mean()

# Calculate team-specific average corners and cards at home
home_secondary = df.groupby('HomeTeam').agg(
    hc_for=('HC', 'mean'),
    hc_against=('AC', 'mean'),
    hy_for=('HY', 'mean'),
    hy_against=('AY', 'mean')
).reset_index()

# Calculate team-specific average corners and cards away
away_secondary = df.groupby('AwayTeam').agg(
    ac_for=('AC', 'mean'),
    ac_against=('HC', 'mean'),
    ay_for=('AY', 'mean'),
    ay_against=('HY', 'mean')
).reset_index()

# Merge the secondary stats back into the main dataframe
df = df.merge(home_secondary, on='HomeTeam', how='left')
df = df.merge(away_secondary, on='AwayTeam', how='left')

# Calculate Expected Corners for the specific match
df['exp_home_corners'] = (df['hc_for'] / league_hc_avg) * (df['ac_against'] / league_hc_avg) * league_hc_avg
df['exp_away_corners'] = (df['ac_for'] / league_ac_avg) * (df['hc_against'] / league_ac_avg) * league_ac_avg

# Calculate Expected Yellow Cards for the specific match
df['exp_home_yellows'] = (df['hy_for'] / league_hy_avg) * (df['ay_against'] / league_hy_avg) * league_hy_avg
df['exp_away_yellows'] = (df['ay_for'] / league_ay_avg) * (df['hy_against'] / league_ay_avg) * league_ay_avg

# Clean up the intermediate columns used for calculation
df = df.drop(columns=[
    'hc_for', 'hc_against', 'hy_for', 'hy_against',
    'ac_for', 'ac_against', 'ay_for', 'ay_against'
])

# Save the updated dataframe
df.to_csv("turkey_advanced_metrics.csv", index=False)
print("Dixon-Coles adjustment and secondary metrics saved to turkey_advanced_metrics.csv")
