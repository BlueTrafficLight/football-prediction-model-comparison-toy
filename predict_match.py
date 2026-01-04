import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from model_comparison import (
    load_matches,
    predict_elo,
    predict_article,
)

from models_xgb_market import (
    build_rolling_features,
    extract_market_probs,
    blend_model_market,
)

# ==========================================================
# XGBoost configuration (shared, consistent)
# ==========================================================

XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    n_estimators=600,
    max_depth=4,
    learning_rate=0.03,
    min_child_weight=15,
    gamma=0.7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=0.4,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1,
)


def _train_xgb(X, y):
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)
    return model


def _format_probs(p):
    p = np.asarray(p, dtype=float)
    p = p / p.sum()  # safety normalisation
    return {
        "HomeWin": float(p[0]),
        "Draw": float(p[1]),
        "AwayWin": float(p[2]),
    }


# ==========================================================
# MAIN PREDICTION FUNCTION
# ==========================================================

def predict_match(
    home_team: str,
    away_team: str,
    match_date,
    df_all: pd.DataFrame,
    use_market: bool = True,
):
    """
    Predict outcome probabilities for an upcoming football match.

    Uses ONLY data strictly before match_date.
    """

    # -----------------------------
    # Sanity checks
    # -----------------------------
    if home_team == away_team:
        raise ValueError("Home and away teams must be different.")

    match_date = pd.to_datetime(match_date)

    teams = set(df_all["HomeTeam"]).union(df_all["AwayTeam"])
    if home_team not in teams or away_team not in teams:
        raise ValueError("One or both teams not found in historical data.")

    # -----------------------------
    # Historical data only
    # -----------------------------
    df_hist = df_all[df_all["Date"] < match_date].copy()
    if len(df_hist) < 50:
        raise ValueError("Not enough historical data before match date.")

    # -----------------------------
    # Output container
    # -----------------------------
    out = {
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "Date": str(match_date.date()),
        "predictions": {},
    }

    # -----------------------------
    # Upcoming match row
    # -----------------------------
    match_row = pd.DataFrame([{
        "Date": match_date,
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "FTHG": np.nan,
        "FTAG": np.nan,
        "y": np.nan,
    }])

    # ======================================================
    # 1) ARTICLE MODEL
    # ======================================================
    try:
        _, _, p = predict_article(df_hist, match_row)
        out["predictions"]["Article"] = _format_probs(p[-1])
    except Exception as e:
        out["predictions"]["Article"] = f"Unavailable ({e})"

    # ======================================================
    # 2) ELO MODEL
    # ======================================================
    try:
        _, _, p = predict_elo(df_hist, match_row)
        out["predictions"]["Elo"] = _format_probs(p[-1])
    except Exception as e:
        out["predictions"]["Elo"] = f"Unavailable ({e})"

    # ======================================================
    # 3) XGBOOST (feature-rich)
    # ======================================================
    try:
        df_aug = pd.concat([df_hist, match_row], ignore_index=True)

        feat = build_rolling_features(df_aug)

        if feat.empty:
            raise ValueError("No rolling features available for this match.")

        # The last feature row MUST correspond to the future match
        feat_match = feat.iloc[-1]

        train_feat = feat.iloc[:-1]
        if train_feat.empty:
            raise ValueError("No training rows for XGBoost.")

        feature_cols = [
            c for c in train_feat.columns
            if c not in ("y", "__row_index__", "Date")
        ]

        X_train = train_feat[feature_cols].to_numpy()
        y_train = train_feat["y"].astype(int).to_numpy()
        X_pred = feat_match[feature_cols].to_numpy().reshape(1, -1)

        model = _train_xgb(X_train, y_train)
        p_xgb = model.predict_proba(X_pred)[0]

        out["predictions"]["XGBoost"] = _format_probs(p_xgb)

    except Exception as e:
        out["predictions"]["XGBoost"] = f"Unavailable ({e})"
        p_xgb = None

    # ======================================================
    # 4) HYBRID (XGB + Market)
    # ======================================================
    if use_market and p_xgb is not None:
        try:
            mkt = extract_market_probs(df_aug).iloc[-1].to_numpy()
            if not np.isnan(mkt).any():
                p_hybrid = blend_model_market(
                    p_model=p_xgb.reshape(1, -1),
                    p_market=mkt.reshape(1, -1),
                    alpha=0.7,
                )[0]
                out["predictions"]["Hybrid"] = _format_probs(p_hybrid)
            else:
                out["predictions"]["Hybrid"] = "No odds available"
        except Exception as e:
            out["predictions"]["Hybrid"] = f"Unavailable ({e})"

    return out
