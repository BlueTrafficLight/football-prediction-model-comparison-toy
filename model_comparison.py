"""
Full football model comparison pipeline with:
- Train/test split by date
- Article GD-6 model
- Elo (draw-calibrated)
- Poisson attack/defence
- XGBoost rolling features
- Diebold–Mariano tests
"""

import os
import glob
import math
import warnings
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import t
from xgboost import XGBClassifier

from models_xgb_market import (
    train_xgb,
    extract_market_probs,
    blend_model_market,
)


# ======================================================
# LOAD + PREP
# ======================================================

OUTCOME_MAP = {"H": 0, "D": 1, "A": 2}

def load_matches(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["y"] = df["FTR"].map(OUTCOME_MAP)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def train_test_split_by_date(df, test_start="2020-08-01"):
    test_start = pd.to_datetime(test_start)
    return df[df["Date"] < test_start], df[df["Date"] >= test_start]


# ======================================================
# Model 1: GOAL DIFFERENCE LAST Mathces
# ======================================================

def article_probs(r):
    h = 1.56 * r + 46.47
    a = 0.03 * r**2 - 1.27 * r + 23.65
    d = -0.03 * r**2 - 0.29 * r + 29.48
    p = np.clip(np.array([h, d, a]) / 100, 1e-6, 1)
    return p / p.sum()


def compute_last6_gd(df):
    hist = {}
    ratings = []

    for _, row in df.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]
        hist.setdefault(ht, [])
        hist.setdefault(at, [])

        def gd(team):
            if len(hist[team]) < 6:
                return np.nan
            g = hist[team][-6:]
            return sum(x[0] - x[1] for x in g)

        ratings.append(gd(ht) - gd(at) if not np.isnan(gd(ht)) and not np.isnan(gd(at)) else np.nan)

        hist[ht].append((row["FTHG"], row["FTAG"]))
        hist[at].append((row["FTAG"], row["FTHG"]))

    df["article_rating"] = ratings
    return df


def predict_article(df_train, df_test):
    df_all = pd.concat([df_train, df_test]).reset_index(drop=True)
    df_all = compute_last6_gd(df_all)

    test_idx = df_all.index[len(df_train):]
    sub = df_all.loc[test_idx].dropna(subset=["article_rating"])

    probs = np.vstack([article_probs(r) for r in sub["article_rating"]])
    return sub.index.values, sub["y"].values, probs


# ======================================================
# Model 2: ELO MODEL
# ======================================================

def elo_predict(df, home_adv=5, k=20, d0=0.28, dscale=250):
    ratings = {}
    probs, y = [], []

    for _, r in df.iterrows():
        ht, at = r["HomeTeam"], r["AwayTeam"]
        ratings.setdefault(ht, 1500)
        ratings.setdefault(at, 1500)

        diff = (ratings[ht] + home_adv) - ratings[at]
        p_home = 1 / (1 + 10 ** (-diff / 400))
        p_draw = d0 * math.exp(-abs(diff) / dscale)
        p_away = 1 - p_home

        p = np.array([(1 - p_draw) * p_home, p_draw, (1 - p_draw) * p_away])
        p /= p.sum()

        probs.append(p)
        y.append(r["y"])

        s = 1 if r["FTR"] == "H" else 0.5 if r["FTR"] == "D" else 0
        ratings[ht] += k * (s - p_home)
        ratings[at] -= k * (s - p_home)

    return np.array(y), np.vstack(probs)


def predict_elo(df_train, df_test):
    df_all = pd.concat([df_train, df_test])
    y_all, p_all = elo_predict(df_all)
    return df_test.index.values, y_all[-len(df_test):], p_all[-len(df_test):]


# ======================================================
# Model 3: POISSON MODEL
# ======================================================

def fit_poisson(df):
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    H = df["HomeTeam"].map(idx)
    A = df["AwayTeam"].map(idx)
    hg, ag = df["FTHG"], df["FTAG"]

    def loss(x):
        mu, ha = x[:2]
        att, dfn = x[2:2+n], x[2+n:]
        lh = np.exp(mu + ha + att[H] - dfn[A])
        la = np.exp(mu + att[A] - dfn[H])
        return -(hg*np.log(lh) - lh + ag*np.log(la) - la).sum()

    res = minimize(loss, np.zeros(2 + 2*n), method="L-BFGS-B")
    mu, ha = res.x[:2]
    att, dfn = res.x[2:2+n], res.x[2+n:]
    return mu, ha, att, dfn, idx


def poisson_probs(mu, ha, att, dfn, idx, home, away, max_goals=10):
    i, j = idx[home], idx[away]
    lh = np.exp(mu + ha + att[i] - dfn[j])
    la = np.exp(mu + att[j] - dfn[i])

    ph = np.array([math.exp(-lh) * lh**k / math.factorial(k) for k in range(max_goals+1)])
    pa = np.array([math.exp(-la) * la**k / math.factorial(k) for k in range(max_goals+1)])
    M = np.outer(ph, pa)

    return np.array([np.triu(M,1).sum(), np.trace(M), np.tril(M,-1).sum()])


def predict_poisson(df_train, df_test, min_matches=200):
    df_all = pd.concat([df_train, df_test]).sort_values("Date").reset_index(drop=True)

    probs, y, idxs = [], [], []

    df_all["Season"] = df_all["Date"].dt.year

    for season, season_df in df_all.groupby("Season"):
        hist = df_all[df_all["Date"] < season_df["Date"].min()]
        if len(hist) < min_matches:
            continue

        mu, ha, att, dfn, idx = fit_poisson(hist)

        for i, row in season_df.iterrows():
            if row["HomeTeam"] not in idx or row["AwayTeam"] not in idx:
                continue

            p = poisson_probs(mu, ha, att, dfn, idx, row["HomeTeam"], row["AwayTeam"])
            probs.append(p)
            y.append(row["y"])
            idxs.append(i)

    return np.array(idxs), np.array(y), np.vstack(probs)


# ======================================================
# Model 4: XGBOOST MODEL
# ======================================================

def build_xgb_features(df, window=6):
    df = df.sort_values("Date").reset_index(drop=True)

    teams = set(df["HomeTeam"]) | set(df["AwayTeam"])
    hist = {t: [] for t in teams}

    rows = []

    for _, r in df.iterrows():
        ht, at = r["HomeTeam"], r["AwayTeam"]

        if len(hist[ht]) >= window and len(hist[at]) >= window:
            rows.append({
                "Date": r["Date"],
                "y": r["y"],
                "home_gf": np.mean([x[0] for x in hist[ht][-window:]]),
                "home_ga": np.mean([x[1] for x in hist[ht][-window:]]),
                "away_gf": np.mean([x[0] for x in hist[at][-window:]]),
                "away_ga": np.mean([x[1] for x in hist[at][-window:]]),
            })

        hist[ht].append((r["FTHG"], r["FTAG"]))
        hist[at].append((r["FTAG"], r["FTHG"]))

    return pd.DataFrame(rows)


def predict_xgb(df):
    feat_df = build_xgb_features(df)

    if len(feat_df) < 500:
        raise RuntimeError("Not enough data for XGBoost after feature construction")

    train, test = train_test_split_by_date(feat_df)

    if train.empty or test.empty:
        raise RuntimeError("Empty train or test set for XGBoost")

    X_train = train[["home_gf", "home_ga", "away_gf", "away_ga"]]
    y_train = train["y"]

    X_test = test[["home_gf", "home_ga", "away_gf", "away_ga"]]
    y_test = test["y"]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return test.index.values, y_test.values, model.predict_proba(X_test)


# ======================================================
# METRICS + DM TEST
# ======================================================

def log_loss(y, p):
    p = np.clip(p, 1e-12, 1)
    return -np.log(p[np.arange(len(y)), y])


def dm_test(l1, l2):
    d = l1 - l2
    dm = d.mean() / np.sqrt(np.var(d, ddof=1) / len(d))
    p = 2 * (1 - t.cdf(abs(dm), df=len(d)-1))
    return dm, p


# ======================================================
# RUN EVERYTHING
# ======================================================

def run():
    # --------------------------------------------------
    # Load & prepare data
    # --------------------------------------------------
    df = load_matches("data/all_seasons.csv") # Use available database

    train, test = train_test_split_by_date(df)

    # Boolean masks over FULL df (needed for XGB module)
    train_mask = (df["Date"] < test["Date"].min()).to_numpy()
    test_mask = ~train_mask

    preds = {}

    # --------------------------------------------------
    # Baseline models (unchanged)
    # --------------------------------------------------
    preds["Article"] = predict_article(train, test)
    preds["Elo"] = predict_elo(train, test)
    preds["Poisson"] = predict_poisson(train, test)

    # --------------------------------------------------
    # Feature-rich XGBoost (NO market odds)
    # --------------------------------------------------
    idx_xgb, y_xgb, p_xgb = train_xgb(
        df_all=df,
        train_mask=train_mask,
        test_mask=test_mask,
        window=10,
        use_market_features=False,
    )
    preds["XGBoost(FeatureRich)"] = (idx_xgb, y_xgb, p_xgb)

    # --------------------------------------------------
    # XGBoost with market odds as FEATURES
    # --------------------------------------------------
    idx_xgb_m, y_xgb_m, p_xgb_m = train_xgb(
        df_all=df,
        train_mask=train_mask,
        test_mask=test_mask,
        window=10,
        use_market_features=True,
    )
    preds["XGBoost+MarketFeatures"] = (idx_xgb_m, y_xgb_m, p_xgb_m)

    # --------------------------------------------------
    # Hybrid: model–market BLEND (industry standard)
    # --------------------------------------------------
    mkt = extract_market_probs(df)
    mkt_test = mkt.loc[idx_xgb_m].to_numpy()

    good = ~np.isnan(mkt_test).any(axis=1)
    p_blend = blend_model_market(
        p_model=p_xgb_m[good],
        p_market=mkt_test[good],
        alpha=0.7,     # can be tuned later
    )

    preds["Hybrid(XGB+MarketBlend)"] = (
        idx_xgb_m[good],
        y_xgb_m[good],
        p_blend,
    )

    # --------------------------------------------------
    # Align predictions (UNCHANGED)
    # --------------------------------------------------
    common = set.intersection(*[set(v[0]) for v in preds.values()])
    preds = {
        k: (
            i[np.isin(i, list(common))],
            y[np.isin(i, list(common))],
            p[np.isin(i, list(common))],
        )
        for k, (i, y, p) in preds.items()
    }

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("\n=== LOG LOSS SUMMARY ===")
    for k, (_, y, p) in preds.items():
        print(f"{k:28s} n={len(y):5d}  logloss={log_loss(y, p).mean():.4f}")

    # --------------------------------------------------
    # Diebold–Mariano tests
    # --------------------------------------------------
    print("\n=== DIEBOLD–MARIANO TESTS ===")
    keys = list(preds.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            l1 = log_loss(preds[keys[i]][1], preds[keys[i]][2])
            l2 = log_loss(preds[keys[j]][1], preds[keys[j]][2])
            dm, pv = dm_test(l1, l2)
            print(f"{keys[i]} vs {keys[j]}: DM={dm:.3f}, p={pv:.4f}")


if __name__ == "__main__":
    run()
