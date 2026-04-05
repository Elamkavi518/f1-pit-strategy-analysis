"""
============================================================
  F1 Pit Stop Strategy Analysis — How Strategy Affects Results
  Author: Elamkavi518 | GitHub: github.com/Elamkavi518
============================================================

Dependencies:
    pip install fastf1 pandas matplotlib seaborn scikit-learn numpy

Run:
    python pit_strategy_analysis.py
"""

import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings("ignore")

# ── Cache setup ──────────────────────────────────────────────────────────────
CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

OUTPUT_DIR = "pit_strategy_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
YEAR   = 2023
RACE   = "Bahrain"
SESSION_TYPE = "R"   # R = Race

print(f"\n{'='*60}")
print(f"  F1 PIT STOP STRATEGY ANALYSIS")
print(f"  {YEAR} {RACE} Grand Prix")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD SESSION
# ─────────────────────────────────────────────────────────────────────────────
print("⏳ Loading session data (this may take a minute)...")
session = fastf1.get_session(YEAR, RACE, SESSION_TYPE)
session.load()
print("✅ Session loaded!\n")

laps = session.laps.copy()

# Keep only quick laps (remove SC laps, pit-in/out laps)
laps = laps[laps["TrackStatus"] == "1"]          # green flag only
laps = laps[laps["PitInTime"].isna()]             # remove pit-in laps
laps = laps[laps["PitOutTime"].isna()]            # remove pit-out laps
laps = laps[laps["LapTime"].notna()]

# Convert LapTime to seconds
laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

# Filter sane lap times (within 150% of median to remove anomalies)
median_lap = laps["LapTimeSeconds"].median()
laps = laps[laps["LapTimeSeconds"] < median_lap * 1.5]
laps = laps[laps["LapTimeSeconds"] > median_lap * 0.85]

print(f"📊 Total clean laps loaded: {len(laps)}")
print(f"   Drivers: {laps['Driver'].nunique()}")
print(f"   Median lap time: {median_lap:.2f}s\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD PIT STOP SUMMARY PER DRIVER
# ─────────────────────────────────────────────────────────────────────────────
print("🔧 Building pit stop summary per driver...")

pit_laps = session.laps[session.laps["PitInTime"].notna()].copy()
pit_laps["PitInTime_s"] = pit_laps["PitInTime"].dt.total_seconds()
pit_laps["PitOutTime_s"] = pit_laps["PitOutTime"].dt.total_seconds()

# Count stints and pit stops per driver
driver_stints = (
    laps.groupby(["Driver", "Stint"])
    .agg(
        StintLaps=("LapNumber", "count"),
        AvgStintPace=("LapTimeSeconds", "mean"),
        Compound=("Compound", "first"),
    )
    .reset_index()
)

driver_summary = (
    driver_stints.groupby("Driver")
    .agg(
        NumStints=("Stint", "nunique"),
        AvgPace=("AvgStintPace", "mean"),
    )
    .reset_index()
)
driver_summary["NumPitStops"] = driver_summary["NumStints"] - 1

# Merge with results
results = session.results[["Abbreviation", "Position", "FullName", "TeamName"]].copy()
results = results.rename(columns={"Abbreviation": "Driver"})
driver_summary = driver_summary.merge(results, on="Driver", how="left")
driver_summary["Position"] = pd.to_numeric(driver_summary["Position"], errors="coerce")
driver_summary = driver_summary.dropna(subset=["Position"])
driver_summary["Position"] = driver_summary["Position"].astype(int)

print(driver_summary[["Driver", "FullName", "NumPitStops", "AvgPace", "Position"]].to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPOUND PERFORMANCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("🔬 Analyzing tyre compound performance...")

compound_stats = (
    laps.groupby("Compound")
    .agg(
        AvgLapTime=("LapTimeSeconds", "mean"),
        MedianLapTime=("LapTimeSeconds", "median"),
        StdLapTime=("LapTimeSeconds", "std"),
        TotalLaps=("LapTimeSeconds", "count"),
    )
    .reset_index()
    .sort_values("AvgLapTime")
)
print(compound_stats.to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. STINT DEGRADATION MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("📉 Modelling tyre degradation per compound...")

deg_data = laps.groupby(["Compound", "TyreLife"]).agg(
    AvgLapTime=("LapTimeSeconds", "mean")
).reset_index()

# ─────────────────────────────────────────────────────────────────────────────
# 5. ML — Does number of pit stops predict finishing position?
# ─────────────────────────────────────────────────────────────────────────────
print("🤖 Running ML regression: pit stops → finishing position...\n")

ml_df = driver_summary[["NumPitStops", "AvgPace", "Position"]].dropna()

X = ml_df[["NumPitStops", "AvgPace"]].values
y = ml_df["Position"].values

model = LinearRegression()
model.fit(X, y)
score = model.score(X, y)

print(f"   Linear Regression R² score: {score:.3f}")
print(f"   Coefficients:")
print(f"     NumPitStops → {model.coef_[0]:+.3f} positions per extra stop")
print(f"     AvgPace     → {model.coef_[1]:+.3f} positions per second slower")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

COMPOUND_COLORS = {
    "SOFT":   "#FF1E1E",
    "MEDIUM": "#FFD700",
    "HARD":   "#F0F0F0",
    "INTER":  "#39B54A",
    "WET":    "#0067FF",
}

plt.style.use("dark_background")
FIG_BG   = "#0D0D0D"
CARD_BG  = "#1A1A1A"
ACCENT   = "#E8002D"
GRID_COL = "#2A2A2A"

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"   💾 Saved → {path}")

# ── PLOT 1: Pit Stops vs Finishing Position ──────────────────────────────────
print("📊 Plot 1: Pit stops vs Finishing position")
fig, ax = plt.subplots(figsize=(10, 6), facecolor=FIG_BG)
ax.set_facecolor(CARD_BG)

stops_grp = driver_summary.groupby("NumPitStops")["Position"].mean().reset_index()

bars = ax.bar(
    stops_grp["NumPitStops"],
    stops_grp["Position"],
    color=[ACCENT, "#FF6B6B", "#FFB347"],
    edgecolor="#333",
    linewidth=0.8,
    zorder=3,
)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"P{h:.1f}",
            ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")

ax.set_xlabel("Number of Pit Stops", color="white", fontsize=12)
ax.set_ylabel("Avg Finishing Position", color="white", fontsize=12)
ax.set_title(f"Pit Stops vs Finishing Position — {YEAR} {RACE} GP",
             color="white", fontsize=14, fontweight="bold", pad=15)
ax.yaxis.grid(True, color=GRID_COL, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
save_fig("01_pitstops_vs_position.png")

# ── PLOT 2: Compound Average Pace ────────────────────────────────────────────
print("📊 Plot 2: Tyre compound average pace")
fig, ax = plt.subplots(figsize=(10, 6), facecolor=FIG_BG)
ax.set_facecolor(CARD_BG)

compounds = compound_stats["Compound"].tolist()
times     = compound_stats["AvgLapTime"].tolist()
colors    = [COMPOUND_COLORS.get(c, "#888") for c in compounds]

bars = ax.bar(compounds, times, color=colors, edgecolor="#444", linewidth=0.8, zorder=3)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, t + 0.1, f"{t:.2f}s",
            ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")

ax.set_xlabel("Tyre Compound", color="white", fontsize=12)
ax.set_ylabel("Average Lap Time (s)", color="white", fontsize=12)
ax.set_title(f"Tyre Compound Average Pace — {YEAR} {RACE} GP",
             color="white", fontsize=14, fontweight="bold", pad=15)
ax.yaxis.grid(True, color=GRID_COL, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors="white")
baseline = min(times) - 1
ax.set_ylim(baseline, max(times) + 2)
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
save_fig("02_compound_pace.png")

# ── PLOT 3: Tyre Degradation (Lap Time vs Tyre Age) ─────────────────────────
print("📊 Plot 3: Tyre degradation per compound")
fig, ax = plt.subplots(figsize=(12, 6), facecolor=FIG_BG)
ax.set_facecolor(CARD_BG)

for compound in deg_data["Compound"].unique():
    sub = deg_data[deg_data["Compound"] == compound]
    sub = sub[sub["TyreLife"] <= 40]
    color = COMPOUND_COLORS.get(compound, "#888")
    ax.plot(sub["TyreLife"], sub["AvgLapTime"], label=compound,
            color=color, linewidth=2.5, marker="o", markersize=3)

ax.set_xlabel("Tyre Age (Laps)", color="white", fontsize=12)
ax.set_ylabel("Average Lap Time (s)", color="white", fontsize=12)
ax.set_title(f"Tyre Degradation by Compound — {YEAR} {RACE} GP",
             color="white", fontsize=14, fontweight="bold", pad=15)
ax.legend(facecolor="#222", edgecolor="#444", labelcolor="white", fontsize=10)
ax.yaxis.grid(True, color=GRID_COL, zorder=0)
ax.xaxis.grid(True, color=GRID_COL, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
save_fig("03_tyre_degradation.png")

# ── PLOT 4: Driver Stint Map (Strategy Visualization) ────────────────────────
print("📊 Plot 4: Driver strategy stint map")

top_drivers = driver_summary.sort_values("Position").head(10)["Driver"].tolist()
stint_data  = driver_stints[driver_stints["Driver"].isin(top_drivers)]

fig, ax = plt.subplots(figsize=(14, 7), facecolor=FIG_BG)
ax.set_facecolor(CARD_BG)

y_positions = {drv: i for i, drv in enumerate(top_drivers)}

for _, row in stint_data.iterrows():
    drv     = row["Driver"]
    stint   = row["Stint"]
    compound= row["Compound"]
    laps_cnt= row["StintLaps"]
    color   = COMPOUND_COLORS.get(compound, "#888")
    y       = y_positions.get(drv, 0)

    # Calculate start lap of this stint (approximate)
    prev_stints = stint_data[(stint_data["Driver"] == drv) & (stint_data["Stint"] < stint)]
    start_lap = prev_stints["StintLaps"].sum()

    ax.barh(y, laps_cnt, left=start_lap, height=0.6,
            color=color, edgecolor="#111", linewidth=0.5, zorder=3)

ax.set_yticks(list(y_positions.values()))
ax.set_yticklabels(list(y_positions.keys()), color="white", fontsize=10)
ax.set_xlabel("Lap Number", color="white", fontsize=12)
ax.set_title(f"Tyre Strategy Map (Top 10) — {YEAR} {RACE} GP",
             color="white", fontsize=14, fontweight="bold", pad=15)
ax.xaxis.grid(True, color=GRID_COL, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors="white")

legend_patches = [mpatches.Patch(color=v, label=k) for k, v in COMPOUND_COLORS.items()
                  if k in laps["Compound"].unique()]
ax.legend(handles=legend_patches, facecolor="#222", edgecolor="#444",
          labelcolor="white", fontsize=9, loc="lower right")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
save_fig("04_strategy_map.png")

# ── PLOT 5: Avg Pace vs Finishing Position scatter ───────────────────────────
print("📊 Plot 5: Average pace vs Finishing position")
fig, ax = plt.subplots(figsize=(10, 6), facecolor=FIG_BG)
ax.set_facecolor(CARD_BG)

scatter = ax.scatter(
    driver_summary["AvgPace"],
    driver_summary["Position"],
    c=driver_summary["NumPitStops"],
    cmap="RdYlGn_r",
    s=120, edgecolors="white", linewidths=0.5, zorder=4
)
for _, row in driver_summary.iterrows():
    ax.annotate(row["Driver"],
                (row["AvgPace"], row["Position"]),
                textcoords="offset points", xytext=(6, 2),
                color="white", fontsize=7, alpha=0.85)

cb = plt.colorbar(scatter, ax=ax)
cb.set_label("Pit Stops", color="white", fontsize=10)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

ax.set_xlabel("Average Lap Time (s)", color="white", fontsize=12)
ax.set_ylabel("Finishing Position", color="white", fontsize=12)
ax.set_title(f"Pace vs Position (colored by pit stops) — {YEAR} {RACE} GP",
             color="white", fontsize=14, fontweight="bold", pad=15)
ax.invert_yaxis()
ax.yaxis.grid(True, color=GRID_COL, zorder=0)
ax.xaxis.grid(True, color=GRID_COL, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
save_fig("05_pace_vs_position.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"\n📁 All plots saved to: ./{OUTPUT_DIR}/")
print("\n  Files generated:")
print("   01_pitstops_vs_position.png")
print("   02_compound_pace.png")
print("   03_tyre_degradation.png")
print("   04_strategy_map.png")
print("   05_pace_vs_position.png")

print(f"\n📌 Key Insights ({YEAR} {RACE} GP):")
best_stop = stops_grp.loc[stops_grp["Position"].idxmin(), "NumPitStops"]
print(f"   ✅ Optimal pit stops: {int(best_stop)} stop(s)")
fastest_compound = compound_stats.iloc[0]["Compound"]
print(f"   🔴 Fastest compound: {fastest_compound}")
print(f"   📈 ML R² (pit stops + pace → position): {score:.3f}")
print(f"   💡 Each extra pit stop shifts position: {model.coef_[0]:+.2f} places")
print()
print("🚀 Upload the output folder images to your GitHub repo!")
print("   LinkedIn post tip: Use the strategy map + pitstops chart as carousel slides.")
print()
