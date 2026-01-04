import streamlit as st
import pandas as pd

from predict_match import predict_match
from model_comparison import load_matches

# ----------------------------------------
# App config
# ----------------------------------------
st.set_page_config(
    page_title="Football Match Predictor",
    layout="centered",
)

st.title("Football Match Predictor")
st.write(
    "Predict match outcome probabilities using multiple models "
    "(Article, Elo, XGBoost, Hybrid)."
)

# ----------------------------------------
# Load data
# ----------------------------------------
@st.cache_data
def load_data():
    return load_matches("data/all_seasons.csv")

df_all = load_data()

# ----------------------------------------
# Sidebar inputs
# ----------------------------------------
st.sidebar.header("Match details")

teams = sorted(pd.unique(df_all[["HomeTeam", "AwayTeam"]].values.ravel()))

home_team = st.sidebar.selectbox("Home team", teams)
away_team = st.sidebar.selectbox(
    "Away team",
    teams,
    index=1 if teams[0] == home_team else 0,
)

match_date = st.sidebar.date_input(
    "Match date",
    value=pd.Timestamp.today(),
)

use_market = st.sidebar.checkbox(
    "Use market odds (if available)",
    value=True,
)

# ----------------------------------------
# Predict button
# ----------------------------------------
if st.sidebar.button("Predict match"):
    if home_team == away_team:
        st.error("Home and away teams must be different.")
    else:
        with st.spinner("Generating predictions..."):
            try:
                result = predict_match(
                    home_team=home_team,
                    away_team=away_team,
                    match_date=match_date,
                    df_all=df_all,
                    use_market=use_market,
                )

                st.subheader(
                    f"{result['HomeTeam']} vs {result['AwayTeam']} "
                    f"({result['Date']})"
                )

                # ----------------------------------------
                # Display predictions
                # ----------------------------------------
                for model, probs in result["predictions"].items():
                    st.markdown(f"### {model}")

                    if isinstance(probs, dict):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Home win", f"{probs['HomeWin']:.2%}")
                        col2.metric("Draw", f"{probs['Draw']:.2%}")
                        col3.metric("Away win", f"{probs['AwayWin']:.2%}")
                    else:
                        st.write(probs)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
