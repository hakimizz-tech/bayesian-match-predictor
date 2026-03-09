import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

st.set_page_config(page_title="Bayesian Match Preview", layout="wide")


@st.cache_data
def load_data():
    # Load the data generated from our previous data science pipeline
    df = pd.read_csv("data/turkey_final_bayesian_model.csv")
    # Ensure Date is parsed correctly for the time-series form tracking
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
teams = sorted(df['HomeTeam'].dropna().unique())


st.title("Interactive Match Preview")
st.markdown("*Applying Storytelling with Data principles to Bayesian Football Predictions.*")

# Create interactive dropdowns for the user to select the matchup
col_h, col_vs, col_a = st.columns([2, 1, 2])
with col_h:
    home_team = st.selectbox("Select Home Team", teams, index=teams.index("Sivasspor") if "Sivasspor" in teams else 0)
with col_vs:
    st.markdown("<h2 style='text-align: center; color: gray; margin-top: 20px;'>VS</h2>", unsafe_allow_html=True)
with col_a:
    away_team = st.selectbox("Select Away Team", teams, index=teams.index("Alanyaspor") if "Alanyaspor" in teams else 1)

# Filter the dataset for the latest match between these two teams
match_history = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)]

if match_history.empty:
    st.warning(f"No historical data found for {home_team} vs {away_team} in this dataset.")
else:
    # Grab the most recent matchup
    match = match_history.iloc[-1]
    
    st.markdown("---")
    
    
   
    # Create a 3-column layout for our visual story
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. The Outcome")
        st.markdown("*Bayesian Win Probabilities*")
        
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        probs = [match['Post_P(h)'], match['Post_P(d)'], match['Post_P(A)']]
        labels = [f"{home_team} Win", "Draw", f"{away_team} Win"]
        colors = ['#1f77b4', '#cccccc', '#ff7f0e'] # Blue, Gray, Orange
        
        ax1.barh(labels, probs, color=colors, height=0.5)
        ax1.set_xlim(0, 1)
        ax1.invert_yaxis()
        
        # Remove all borders and axes
        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.tick_params(left=False, bottom=False, labelbottom=False)
        
        # Put text directly next to bars
        for i, p in enumerate(probs):
            ax1.text(p + 0.02, i, f"{p*100:.1f}%", va='center', fontsize=12, fontweight='bold', color=colors[i])
            
        st.pyplot(fig1)

    with col2:
        st.subheader("2. The Details")
        st.markdown("*Most Likely Exact Scorelines (%)*")
        
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        max_goals = 4 
        home_probs = poisson.pmf(np.arange(max_goals), match['lambda h'])
        away_probs = poisson.pmf(np.arange(max_goals), match['lambda a'])
        prob_matrix = np.outer(home_probs, away_probs) * 100 
        
        # USE PRE-ATTENTIVE ATTRIBUTES: Heatmap color saturation draws the eye to the highest probability
        sns.heatmap(prob_matrix, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=ax2, 
                    annot_kws={"size": 11, "weight": "bold"})
        
        ax2.set_xlabel(f"{away_team} Goals (Away)", fontsize=10, color='gray')
        ax2.set_ylabel(f"{home_team} Goals (Home)", fontsize=10, color='gray')
        
        st.pyplot(fig2)

    # COMPARATIVE STRENGTHS 
    with col3:
        st.subheader("3. The 'Why'")
        st.markdown("*Team Strengths vs League Average*")
        
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        metrics = ['Attack Strength\n(Higher is better)', 'Defense Fragility\n(Lower is better)']
        home_stats = [match['H attack'], match['H def']]
        away_stats = [match['A attack'], match['A def']]
        
        y = np.arange(len(metrics))
        height = 0.3
        
        ax3.barh(y - height/2, home_stats, height, label=home_team, color='#1f77b4')
        ax3.barh(y + height/2, away_stats, height, label=away_team, color='#ff7f0e')
        
        # CONTEXT: Add a subtle gray baseline for the League Average
        ax3.axvline(1.0, color='#cccccc', linestyle='--', linewidth=1.5, zorder=0)
        ax3.text(1.02, -0.4, "League Avg", color='gray', fontsize=9, va='center')
        
        # DECLUTTERING
        for spine in ax3.spines.values():
            spine.set_visible(False)
        ax3.tick_params(left=False, bottom=False, labelbottom=False)
        ax3.set_yticks(y)
        ax3.set_yticklabels(metrics, fontsize=10, color='#333333')
        ax3.invert_yaxis()
        
        # DIRECT LABELING
        for i, stat in enumerate(home_stats):
            ax3.text(stat + 0.05, i - height/2, f"{stat:.2f}", va='center', color='#1f77b4', fontweight='bold')
        for i, stat in enumerate(away_stats):
            ax3.text(stat + 0.05, i + height/2, f"{stat:.2f}", va='center', color='#ff7f0e', fontweight='bold')
            
        st.pyplot(fig3)

    st.markdown("---")

    
    # THE ANALYTICAL DEEP DIVE
    
    col4, col5 = st.columns(2)

    # 4. SLOPEGRAPH (The Bayesian Before and After)
    with col4:
        st.subheader("4. The Bayesian Impact")
        st.markdown("*How historical Head-to-Head records shifted the base probabilities.*")
        
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        
        # Coordinates for Left (Base Poisson) and Right (Bayesian Posterior)
        x = [0, 1]
        y_home = [match['P(h)'], match['Post_P(h)']]
        y_draw = [match['P(d)'], match['Post_P(d)']]
        y_away = [match['P(A)'], match['Post_P(A)']]
        
        # Plot lines
        ax4.plot(x, y_home, color='#1f77b4', marker='o', markersize=8, linewidth=3)
        ax4.plot(x, y_draw, color='#cccccc', marker='o', markersize=8, linewidth=3)
        ax4.plot(x, y_away, color='#ff7f0e', marker='o', markersize=8, linewidth=3)
        
        # Direct Labeling Left Side (Base)
        ax4.text(-0.05, y_home[0], f"{home_team} ({y_home[0]*100:.1f}%)", ha='right', va='center', color='#1f77b4', fontweight='bold')
        ax4.text(-0.05, y_draw[0], f"Draw ({y_draw[0]*100:.1f}%)", ha='right', va='center', color='gray', fontweight='bold')
        ax4.text(-0.05, y_away[0], f"{away_team} ({y_away[0]*100:.1f}%)", ha='right', va='center', color='#ff7f0e', fontweight='bold')
        
        # Direct Labeling Right Side (Updated)
        ax4.text(1.05, y_home[1], f"{y_home[1]*100:.1f}%", ha='left', va='center', color='#1f77b4', fontweight='bold')
        ax4.text(1.05, y_draw[1], f"{y_draw[1]*100:.1f}%", ha='left', va='center', color='gray', fontweight='bold')
        ax4.text(1.05, y_away[1], f"{y_away[1]*100:.1f}%", ha='left', va='center', color='#ff7f0e', fontweight='bold')
        
        # Declutter and format axes
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(["Base Poisson\n(Season Form)", "Bayesian Model\n(H2H Adjusted)"], fontweight='bold', color='#333333')
        ax4.set_xlim(-0.5, 1.5)
        
        for spine in ax4.spines.values():
            spine.set_visible(False)
        ax4.set_yticks([])
        ax4.tick_params(bottom=False, left=False)
        
        st.pyplot(fig4)


    with col5:
        st.subheader("5. Form Tracking")
        st.markdown("*Expected Goals (xG) generated over the last 10 matches.*")
        
        # Extract the last 10 matches where the home team played at home, and away team played away
        home_past = df[df['HomeTeam'] == home_team].tail(10)
        away_past = df[df['AwayTeam'] == away_team].tail(10)
        
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        
        # Plot continuous data
        ax5.plot(range(1, len(home_past) + 1), home_past['lambda h'], color='#1f77b4', linewidth=2.5, marker='o')
        ax5.plot(range(1, len(away_past) + 1), away_past['lambda a'], color='#ff7f0e', linewidth=2.5, marker='o')
        
        # Direct labeling at the end of the line (No legends!)
        if not home_past.empty:
            ax5.text(len(home_past) + 0.2, home_past['lambda h'].iloc[-1], home_team, color='#1f77b4', fontweight='bold', va='center')
        if not away_past.empty:
            ax5.text(len(away_past) + 0.2, away_past['lambda a'].iloc[-1], away_team, color='#ff7f0e', fontweight='bold', va='center')
            
        # Decluttering 
        for spine in ['top', 'right', 'left']:
            ax5.spines[spine].set_visible(False)
        
        ax5.spines['bottom'].set_color('#cccccc')
        ax5.tick_params(left=False, colors='gray')
        ax5.set_xticks([1, 5, 10])
        ax5.set_xticklabels(['Older Matches', '', 'Recent Matches'])
        
        # Subtle horizontal gridlines to help trace values, but kept very faint
        ax5.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        st.pyplot(fig5)
