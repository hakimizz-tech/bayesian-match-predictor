import pandas as pd
import glob

# Find all the downloaded CSVs
files = glob.glob("turkey_*.csv")

# Read and combine them into a single DataFrame
df_list = [pd.read_csv(file) for file in files]
master_df = pd.concat(df_list, ignore_index=True)

clean_df = master_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HC', 'AC', 'HY', 'AY', 'HS', 'AS']].copy()

# Sort chronologically just to be safe
clean_df['Date'] = pd.to_datetime(clean_df['Date'], dayfirst=True)
clean_df = clean_df.sort_values(by='Date').reset_index(drop=True)

# Save your new, massive, clean dataset!
clean_df.to_csv("turkey_master_clean.csv", index=False)
print("Data combined successfully!")
