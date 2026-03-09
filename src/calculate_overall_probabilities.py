import pandas as pd
import numpy as np
from scipy.stats import poisson

def calculate_match_probabilities(row, max_goals=10):
    """
    Generates a matrix of scoreline probabilities and sums them to find 
    the overall likelihood of a Home Win, Draw, or Away Win.
    """
    home_expected = row['lambda h']
    away_expected = row['lambda a']
    
    # Generate arrays of probabilities for scoring 0 through max_goals
    goals_range = np.arange(max_goals)
    home_probs = poisson.pmf(goals_range, home_expected)
    away_probs = poisson.pmf(goals_range, away_expected)
    
    # Create the bivariate probability matrix (home goals on rows, away goals on columns)
    prob_matrix = np.outer(home_probs, away_probs)
    
    # Sum the lower triangle where Home Goals (rows) > Away Goals (columns)
    p_home = np.tril(prob_matrix, k=-1).sum()
    
    # Sum the main diagonal where Home Goals == Away Goals
    p_draw = np.trace(prob_matrix)
    
    # Sum the upper triangle where Home Goals < Away Goals
    p_away = np.triu(prob_matrix, k=1).sum()
    
    return pd.Series([p_home, p_draw, p_away])

# Load the dataset from the previous form calculation script
df = pd.read_csv("../data/turkey_form_and_poisson_features.csv")

# Apply the probability calculation across the dataframe
df[['P(h)', 'P(d)', 'P(A)']] = df.apply(calculate_match_probabilities, axis=1)

# Save the dataset with the new overall probabilities
df.to_csv("turkey_overall_probabilities.csv", index=False)
print("Overall match probabilities calculated and saved to turkey_overall_probabilities.csv")
