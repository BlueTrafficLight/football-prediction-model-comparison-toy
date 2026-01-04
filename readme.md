# ⚽ Football Match Prediction (Toy Project)

This is a small **educational / toy project** that implements and compares several **common football match prediction methods**.

The goal is **not to beat bookmakers or provide betting advice**, but to demonstrate how different modelling approaches behave and differ in practice.

---

## 🔍 Models Included

The project compares:

- **Article-style form model**
  - Recent goal-difference based probabilities

- **Elo rating model**
  - Team strength ratings with home advantage

- **XGBoost (feature-based ML)**
  - Rolling pre-match statistics (goals, shots, dominance proxies)
  - Strictly leakage-free

- **Hybrid model**
  - Simple blend of ML and market probabilities (when available)

All models predict **Home / Draw / Away** probabilities.

---

## 📊 What This Project Shows

- How traditional rating models differ from ML-based models
- The impact of home advantage assumptions
- Bias–variance trade-offs in football prediction
- Why different models can give very different probabilities for the same match

---

## 🌐 Streamlit App

A simple **Streamlit web interface** allows users to:
- Select a home team, away team, and match date
- View and compare predictions from each model

Predictions use **only historical data available before the match date**.

---

## ⚠️ Notes

- No live data (lineups, injuries, weather)
- No betting advice
- Results are probabilistic and illustrative only

This project is intended purely for **learning and experimentation**.

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
