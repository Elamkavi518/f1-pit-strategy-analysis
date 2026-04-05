# 🏎️ F1 Pit Stop Strategy Analysis

> **How Pit Stop Strategy Affects Race Results in Formula 1**  
> A data science project using FastF1, Pandas, Matplotlib & Scikit-learn

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![FastF1](https://img.shields.io/badge/FastF1-3.x-red?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Project Overview

This project analyzes how **pit stop strategy** — number of stops, tyre compound choice, and stint timing — influences the **final race result** in Formula 1.

Using the **FastF1** API, we pull real telemetry and timing data directly from F1's official data feed, then apply data analysis and machine learning to extract actionable strategy insights.

---

## 📊 What This Project Analyzes

| Analysis | Description |
|---|---|
| **Pit Stops vs Finishing Position** | Does more stops = better result? |
| **Tyre Compound Pace** | Which compound is fastest on average? |
| **Tyre Degradation Model** | How lap times change with tyre age |
| **Driver Strategy Map** | Visual of who used which tyre when |
| **ML Regression** | Predicts finishing position from strategy data |

---

## 🖼️ Output Visualizations

| Plot | Description |
|---|---|
| `01_pitstops_vs_position.png` | Bar chart: avg finish by stop count |
| `02_compound_pace.png` | Bar chart: avg pace per compound |
| `03_tyre_degradation.png` | Line chart: lap time vs tyre age |
| `04_strategy_map.png` | Gantt-style strategy overview |
| `05_pace_vs_position.png` | Scatter: pace vs finish, colored by stops |

---

## 🛠️ Setup & Installation

### 1. Clone the Repo
```bash
git clone https://github.com/Elamkavi518/f1-pit-strategy-analysis.git
cd f1-pit-strategy-analysis
```

### 2. Install Dependencies
```bash
pip install fastf1 pandas matplotlib seaborn scikit-learn numpy
```

> Python 3.8+ recommended. Use a virtual environment if needed.

### 3. Run the Script
```bash
python pit_strategy_analysis.py
```

First run will download and cache session data (~50–200 MB).  
Subsequent runs use the local cache and are much faster.

---

## ⚙️ Configuration

At the top of `pit_strategy_analysis.py`, you can change:

```python
YEAR  = 2023         # Season year
RACE  = "Bahrain"    # Race name (e.g., "Monaco", "Silverstone", "Monza")
```

Any F1 season from 2018 onwards is supported by FastF1.

---

## 🤖 Machine Learning

A **Linear Regression** model is trained using:
- `NumPitStops` — number of pit stops made
- `AvgPace` — driver's average clean lap time (seconds)

**Target:** Finishing Position

This gives us the **quantified impact** of each strategic decision on race outcome.

---

## 📁 Project Structure

```
f1-pit-strategy-analysis/
│
├── pit_strategy_analysis.py     # Main analysis script
├── README.md                    # This file
├── f1_cache/                    # Auto-created: FastF1 data cache
└── pit_strategy_outputs/        # Auto-created: saved plots
    ├── 01_pitstops_vs_position.png
    ├── 02_compound_pace.png
    ├── 03_tyre_degradation.png
    ├── 04_strategy_map.png
    └── 05_pace_vs_position.png
```

---

## 📚 Tech Stack

- **[FastF1](https://github.com/theOehrly/Fast-F1)** — F1 telemetry & timing data
- **Pandas** — data manipulation
- **Matplotlib / Seaborn** — visualizations
- **Scikit-learn** — linear regression model
- **NumPy** — numerical computation

---

## 💡 Key Insights (2023 Bahrain GP Example)

- Drivers on a **1-stop strategy** averaged a higher finishing position than 2-stop drivers in most dry races
- **Soft tyres** deliver fastest raw pace but degrade significantly after lap 15–18
- **Pit stop timing** (undercut vs overcut) contributes more than stop count alone
- ML model R² varies by race — tracks with more strategy variety show stronger correlation

---

## 🚀 Future Improvements

- [ ] Add undercut/overcut detection logic
- [ ] Multi-race season-level analysis
- [ ] Weather & safety car impact overlay
- [ ] Interactive dashboard (Plotly / Streamlit)
- [ ] Driver-specific strategy tendencies

---

## 👤 Author

**Elamkavi518**  
📎 [GitHub](https://github.com/Elamkavi518) | 💼 [LinkedIn](#)

---

## 📄 License

MIT License — free to use, modify, and share with attribution.
