# 🧠 In-Play Edge Engine  
**ML-Driven Sports Betting Value Detection (NBA + NFL)**  


---

## 📊 Overview

**In-Play Edge Engine** is an end-to-end **machine learning pipeline** that detects *profitable betting opportunities* in real time.  
It connects directly to live sportsbook APIs, processes team and performance stats, and calculates **expected value (EV)** and **edge** for every matchup.

This project showcases **data ingestion**, **modeling**, and **automation** using **Python, Polars, LightGBM, and Rich CLI visualization.**

---

## 🧩 Core Features

| Feature | Description |
|----------|--------------|
| ⚡ **Live Odds Integration** | Fetches odds from [The Odds API](https://the-odds-api.com) (DraftKings, FanDuel). |
| 🧮 **AI Model Calibration** | Uses `GradientBoostingClassifier` + `CalibratedClassifierCV` for accurate win probabilities. |
| 📊 **Value Edge Detection** | Calculates the difference between model probability and market implied probability. |
| 💸 **Expected Value (EV)** | Computes the expected dollar value of a bet given the model’s forecast. |
| 🔁 **Auto Refresh Loop** | Updates odds and model predictions every 15 minutes automatically. |
| 🏀 **Multi-League Support** | NBA and NFL currently implemented (extendable to MLB, NHL, etc.). |
| 🎨 **Rich CLI Output** | Excellent terminal tables with color-coded value indicators. |

---

## 🧠 System Architecture


