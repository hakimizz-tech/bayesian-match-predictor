import pandas as pd

# Load the dataset containing the adjusted probabilities and secondary metrics
df = pd.read_csv("turkey_advanced_metrics.csv")

# Ensure the dataset is sorted chronologically to prevent data leakage in the rolling H2H calculation
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date').reset_index(drop=True)

# Dictionary to store cumulative head-to-head records for each matchup
h2h_records = {}

# Lists to collect the updated posterior probabilities row by row
post_p_h = []
post_p_d = []
post_p_a = []

for index, row in df.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    matchup = (home_team, away_team)
    
    # Initialize the matchup tracking if these teams have not played yet
    if matchup not in h2h_records:
        h2h_records[matchup] = {
            'matches': 0, 
            'h_wins': 0, 
            'draws': 0, 
            'a_wins': 0
        }
        
    # Retrieve the historical head-to-head data prior to this specific match
    history = h2h_records[matchup]
    beta = history['matches']
    obs_h = history['h_wins']
    obs_d = history['draws']
    obs_a = history['a_wins']
    
    # Apply the Gamma-Poisson Bayesian update if historical data exists
    if beta > 0:
        # Calculate alpha (the predicted outcomes based on prior probabilities)
        alpha_h = beta * row['P(h)']
        alpha_d = beta * row['P(d)']
        alpha_a = beta * row['P(A)']
        
        # Calculate alpha prime by adding the actual observed outcomes
        alpha_prime_h = alpha_h + obs_h
        alpha_prime_d = alpha_d + obs_d
        alpha_prime_a = alpha_a + obs_a
        
        # Normalize the alpha prime values to convert them back into probabilities
        total_alpha_prime = alpha_prime_h + alpha_prime_d + alpha_prime_a
        post_p_h.append(alpha_prime_h / total_alpha_prime)
        post_p_d.append(alpha_prime_d / total_alpha_prime)
        post_p_a.append(alpha_prime_a / total_alpha_prime)
    else:
        # Fallback to the base probabilities if there is no historical head-to-head data
        post_p_h.append(row['P(h)'])
        post_p_d.append(row['P(d)'])
        post_p_a.append(row['P(A)'])
        
    # Update the historical records with the current match's outcome for future iterations
    h2h_records[matchup]['matches'] += 1
    h2h_records[matchup]['h_wins'] += row['h win']
    h2h_records[matchup]['draws'] += row['h draw']
    h2h_records[matchup]['a_wins'] += row['a win']

# Assign the final Bayesian posterior probabilities to the dataframe
df['Post_P(h)'] = post_p_h
df['Post_P(d)'] = post_p_d
df['Post_P(A)'] = post_p_a

# Save the finalized predictive dataset
df.to_csv("turkey_final_bayesian_model.csv", index=False)
print("Bayesian posterior probabilities calculated and saved to turkey_final_bayesian_model.csv")
