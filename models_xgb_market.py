"""
Feature-rich Gradient Boosting + Market-informed hybrid models
==============================================================

This module provides:
- Rolling, leakage-safe feature engineering
- Bookmaker odds -> implied probabilities
- XGBoost training (with or without market odds)
- Hybrid model–market probability blending

Designed to plug directly into model_comparison.py
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


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

# ======================================================
# Odds handling
# ======================================================

ODDS_TRIPLETS = [
    ("B365H", "B365D", "B365A"),
    ("PSH", "PSD", "PSA"),
    ("WHH", "WHD", "WHA"),
    ("IWH", "IWD", "IWA"),
    ("LBH", "LBD", "LBA"),
    ("VCH", "VCD", "VCA"),
    ("BWH", "BWD", "BWA"),
    ("SJH", "SJD", "SJA"),
]


def extract_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract implied probabilities from the first available bookmaker odds triplet.
    Returns columns: mkt_pH, mkt_pD, mkt_pA
    """
    out = pd.DataFrame(index=df.index, columns=["mkt_pH", "mkt_pD", "mkt_pA"], dtype=float)

    for h, d, a in ODDS_TRIPLETS:
        if h in df.columns and d in df.columns and a in df.columns:
            oh = pd.to_numeric(df[h], errors="coerce")
            od = pd.to_numeric(df[d], errors="coerce")
            oa = pd.to_numeric(df[a], errors="coerce")

            good = (oh > 1) & (od > 1) & (oa > 1)
            inv = pd.concat([1 / oh, 1 / od, 1 / oa], axis=1)
            s = inv.sum(axis=1)

            out.loc[good, "mkt_pH"] = inv.iloc[:, 0] / s
            out.loc[good, "mkt_pD"] = inv.iloc[:, 1] / s
            out.loc[good, "mkt_pA"] = inv.iloc[:, 2] / s
            break

    return out


# ======================================================
# Rolling pre-match features
# ======================================================

STAT_PAIRS = [
    ("FTHG", "FTAG"),
    ("HS", "AS"),
    ("HST", "AST"),
    ("HC", "AC"),
    ("HF", "AF"),
    ("HY", "AY"),
    ("HR", "AR"),
]


def build_rolling_features(
    df: pd.DataFrame,
    window_short: int = 5,
    window_long: int = 15,
) -> pd.DataFrame:
    """
    Leakage-safe rolling features with MULTIPLE windows.
    Uses goals, shots, shots-on-target, corners if available.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    use_shots = "HS" in df.columns and "AS" in df.columns
    use_sot = "HST" in df.columns and "AST" in df.columns
    use_corners = "HC" in df.columns and "AC" in df.columns

    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel("K"))
    hist = {t: [] for t in teams}

    rows = []

    for i, r in df.iterrows():
        ht, at = r["HomeTeam"], r["AwayTeam"]

        if (
            len(hist[ht]) >= window_long
            and len(hist[at]) >= window_long
        ):
            def avg(team_hist, key, w):
                return np.mean([x[key] for x in team_hist[-w:]])

            feat = {
                "__row_index__": i,
                "Date": r["Date"],
                "y": r["y"],
            }

            # --------------------------------------------------
            # Goals (short + long)
            # --------------------------------------------------
            for w, label in [(window_short, "s"), (window_long, "l")]:
                feat[f"home_gf_{label}"] = avg(hist[ht], "gf", w)
                feat[f"home_ga_{label}"] = avg(hist[ht], "ga", w)
                feat[f"away_gf_{label}"] = avg(hist[at], "gf", w)
                feat[f"away_ga_{label}"] = avg(hist[at], "ga", w)

                feat[f"gf_diff_{label}"] = feat[f"home_gf_{label}"] - feat[f"away_gf_{label}"]
                feat[f"ga_diff_{label}"] = feat[f"home_ga_{label}"] - feat[f"away_ga_{label}"]

            # --------------------------------------------------
            # Shots & SoT (xG proxies)
            # --------------------------------------------------
            if use_shots:
                for w, label in [(window_short, "s"), (window_long, "l")]:
                    feat[f"shot_diff_{label}"] = (
                        avg(hist[ht], "s_for", w) - avg(hist[at], "s_for", w)
                    )

            if use_sot:
                for w, label in [(window_short, "s"), (window_long, "l")]:
                    h_st = avg(hist[ht], "st_for", w)
                    a_st = avg(hist[at], "st_for", w)

                    feat[f"sot_diff_{label}"] = h_st - a_st

                    if use_shots:
                        h_s = avg(hist[ht], "s_for", w)
                        a_s = avg(hist[at], "s_for", w)

                        feat[f"home_sot_rate_{label}"] = h_st / (h_s + 1e-6)
                        feat[f"away_sot_rate_{label}"] = a_st / (a_s + 1e-6)

            # --------------------------------------------------
            # Corners (pressure)
            # --------------------------------------------------
            if use_corners:
                for w, label in [(window_short, "s"), (window_long, "l")]:
                    feat[f"corner_diff_{label}"] = (
                        avg(hist[ht], "c_for", w) - avg(hist[at], "c_for", w)
                    )

            rows.append(feat)

        # --------------------------------------------------
        # Update histories AFTER feature creation
        # --------------------------------------------------
        rec_h = {"gf": r["FTHG"], "ga": r["FTAG"]}
        rec_a = {"gf": r["FTAG"], "ga": r["FTHG"]}

        if use_shots:
            rec_h["s_for"] = r["HS"]
            rec_h["s_ag"] = r["AS"]
            rec_a["s_for"] = r["AS"]
            rec_a["s_ag"] = r["HS"]

        if use_sot:
            rec_h["st_for"] = r["HST"]
            rec_h["st_ag"] = r["AST"]
            rec_a["st_for"] = r["AST"]
            rec_a["st_ag"] = r["HST"]

        if use_corners:
            rec_h["c_for"] = r["HC"]
            rec_h["c_ag"] = r["AC"]
            rec_a["c_for"] = r["AC"]
            rec_a["c_ag"] = r["HC"]

        hist[ht].append(rec_h)
        hist[at].append(rec_a)

    return pd.DataFrame(rows)


# ======================================================
# XGBoost models
# ======================================================

def train_xgb(
    df_all: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    window: int = 10,
    use_market_features: bool = True,
):
    """
    Train XGBoost on rolling features, optionally including market odds.
    Returns (match_indices, y_test, p_test).
    """
    feat = build_rolling_features(
        df_all,
        window_short=5,
        window_long=15,
    )
    mkt = extract_market_probs(df_all)

    feat = feat.merge(
        mkt.assign(__row_index__=mkt.index),
        on="__row_index__",
        how="left",
    )

    feat["is_train"] = train_mask[feat["__row_index__"]]
    feat["is_test"] = test_mask[feat["__row_index__"]]

    feature_cols = [
        c for c in feat.columns
        if (
            c.endswith("_s")
            or c.endswith("_l")
            or c.startswith("home_")
            or c.startswith("away_")
        )
        and c not in ("HomeTeam", "AwayTeam")
    ]

    # Remove non-numeric / target columns explicitly
    feature_cols = [
        c for c in feature_cols
        if c not in ("y", "__row_index__", "Date")
    ]

    if use_market_features:
        feature_cols += ["mkt_pH", "mkt_pD", "mkt_pA"]

    feat = feat.dropna(subset=feature_cols + ["y"])

    train = feat[feat["is_train"]]
    test = feat[feat["is_test"]]

    if train.empty or test.empty:
        raise RuntimeError("XGBoost train/test split produced empty set")

    X_train = train[feature_cols].to_numpy()
    y_train = train["y"].astype(int).to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test["y"].astype(int).to_numpy()

    model = XGBClassifier(**XGB_PARAMS)

    # --------------------------------------------------
    # Time-decay sample weights
    # --------------------------------------------------
    days_ago = (train["Date"].max() - train["Date"]).dt.days

    # 1-year characteristic decay (tuneable)
    tau = 180.0
    sample_weight = np.exp(-days_ago / tau)

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
    )
    p_test = model.predict_proba(X_test)

    return test["__row_index__"].to_numpy(), y_test, p_test


# ======================================================
# Hybrid model–market blend
# ======================================================

def blend_model_market(p_model: np.ndarray, p_market: np.ndarray, alpha: float = 0.7):
    """
    Logit-space blending of model and market probabilities.
    """
    eps = 1e-12
    p_model = np.clip(p_model, eps, 1 - eps)
    p_market = np.clip(p_market, eps, 1 - eps)

    logit = lambda p: np.log(p / (1 - p))
    sigmoid = lambda z: 1 / (1 + np.exp(-z))

    z = alpha * logit(p_model) + (1 - alpha) * logit(p_market)
    p = sigmoid(z)
    return p / p.sum(axis=1, keepdims=True)
