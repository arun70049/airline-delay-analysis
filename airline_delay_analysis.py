# ============================================================
# PROJECT 1: Airline Delay & Customer Satisfaction Analysis
# Author   : Arun Kumar
# Stack    : Python (Pandas, NumPy, Scikit-learn, Matplotlib)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── STEP 1: Generate Realistic Dataset (50,000 rows) ─────────────────────────
print("=" * 60)
print("  Airline Delay & Customer Satisfaction Analysis")
print("=" * 60)
print("\n[1/6] Generating dataset (50,000 rows)...")

n = 50000
routes    = ["DEL-BOM", "BOM-BLR", "DEL-HYD", "BLR-CCU", "HYD-DEL",
             "BOM-CCU", "DEL-BLR", "CCU-DEL", "BLR-DEL", "HYD-BOM"]
carriers  = ["IndiGo", "Air India", "SpiceJet", "Vistara", "GoFirst"]
seasons   = ["Winter", "Summer", "Monsoon", "Autumn"]
delay_causes = ["Weather", "Technical", "ATC", "Crew", "None"]

df = pd.DataFrame({
    "route"         : np.random.choice(routes, n),
    "carrier"       : np.random.choice(carriers, n),
    "season"        : np.random.choice(seasons, n),
    "delay_cause"   : np.random.choice(delay_causes, n, p=[0.20, 0.15, 0.10, 0.10, 0.45]),
    "dep_hour"      : np.random.randint(5, 23, n),
    "flight_duration": np.random.randint(60, 180, n),
    "passengers"    : np.random.randint(80, 200, n),
    "prev_delay_min": np.random.exponential(20, n).astype(int),
})

# Delay in minutes — influenced by cause and season
base_delay = np.where(df["delay_cause"] == "None", 0,
             np.where(df["delay_cause"] == "Weather", np.random.randint(30, 120, n),
             np.where(df["delay_cause"] == "Technical", np.random.randint(20, 90, n),
             np.random.randint(10, 60, n))))

season_add = np.where(df["season"] == "Monsoon", np.random.randint(10, 40, n), 0)
df["delay_minutes"] = base_delay + season_add

# Satisfaction score (1-5) — lower when delay is high
df["satisfaction"] = np.clip(
    5 - (df["delay_minutes"] / 40).astype(int) + np.random.randint(-1, 2, n), 1, 5
)

# Target: delayed = 1 if delay > 15 min
df["is_delayed"] = (df["delay_minutes"] > 15).astype(int)

print(f"   ✓ Dataset created: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── STEP 2: Data Cleaning ─────────────────────────────────────────────────────
print("\n[2/6] Cleaning data...")
before = len(df)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
print(f"   ✓ Removed {before - len(df)} duplicates/nulls")
print(f"   ✓ Clean dataset: {len(df):,} rows")

# ── STEP 3: EDA ───────────────────────────────────────────────────────────────
print("\n[3/6] Exploratory Data Analysis...")

print("\n   --- Delay Rate by Carrier ---")
carrier_delay = df.groupby("carrier")["is_delayed"].mean().sort_values(ascending=False)
for c, v in carrier_delay.items():
    print(f"   {c:<12}: {v*100:.1f}% delayed")

print("\n   --- Avg Delay by Season ---")
season_delay = df.groupby("season")["delay_minutes"].mean().sort_values(ascending=False)
for s, v in season_delay.items():
    print(f"   {s:<10}: {v:.1f} min avg delay")

print("\n   --- Top 3 Delay-Prone Routes ---")
route_delay = df.groupby("route")["delay_minutes"].mean().sort_values(ascending=False).head(3)
for r, v in route_delay.items():
    print(f"   {r}: {v:.1f} min avg delay")

print("\n   --- Satisfaction Distribution ---")
sat_dist = df["satisfaction"].value_counts().sort_index()
for score, count in sat_dist.items():
    bar = "█" * (count // 1000)
    print(f"   ⭐{score}: {bar} ({count:,})")

# ── STEP 4: Visualisations ────────────────────────────────────────────────────
print("\n[4/6] Generating visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Airline Delay & Customer Satisfaction Analysis", fontsize=15, fontweight="bold", color="#1B2A4A")

colors_list = ["#2E6DA4", "#4A9DD4", "#6BB8E8", "#9DD0F0", "#C5E5F8"]

# Plot 1: Delay rate by carrier
ax1 = axes[0, 0]
bars = ax1.bar(carrier_delay.index, carrier_delay.values * 100, color=colors_list)
ax1.set_title("Delay Rate by Carrier (%)", fontweight="bold")
ax1.set_ylabel("Delay Rate (%)")
ax1.set_xticklabels(carrier_delay.index, rotation=20)
for bar, val in zip(bars, carrier_delay.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val*100:.1f}%", ha="center", fontsize=8)

# Plot 2: Avg delay by season
ax2 = axes[0, 1]
bars2 = ax2.bar(season_delay.index, season_delay.values, color=colors_list)
ax2.set_title("Average Delay by Season (min)", fontweight="bold")
ax2.set_ylabel("Avg Delay (minutes)")
for bar, val in zip(bars2, season_delay.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.1f}", ha="center", fontsize=9)

# Plot 3: Delay cause distribution
ax3 = axes[1, 0]
cause_counts = df["delay_cause"].value_counts()
ax3.pie(cause_counts.values, labels=cause_counts.index, autopct="%1.1f%%",
        colors=["#2E6DA4","#4A9DD4","#6BB8E8","#9DD0F0","#C5E5F8"], startangle=90)
ax3.set_title("Delay Cause Distribution", fontweight="bold")

# Plot 4: Satisfaction vs Delay scatter
ax4 = axes[1, 1]
sample = df.sample(500)
ax4.scatter(sample["delay_minutes"], sample["satisfaction"],
            alpha=0.4, color="#2E6DA4", s=15)
ax4.set_title("Delay vs Customer Satisfaction", fontweight="bold")
ax4.set_xlabel("Delay (minutes)")
ax4.set_ylabel("Satisfaction Score (1–5)")

plt.tight_layout()
plt.savefig("/home/claude/airline_eda_charts.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ Charts saved: airline_eda_charts.png")

# ── STEP 5: Predictive Model ──────────────────────────────────────────────────
print("\n[5/6] Building Random Forest Classifier...")

le = LabelEncoder()
df_model = df.copy()
for col in ["route", "carrier", "season", "delay_cause"]:
    df_model[col] = le.fit_transform(df_model[col])

features = ["route", "carrier", "season", "dep_hour", "flight_duration",
            "passengers", "prev_delay_min", "delay_cause"]
X = df_model[features]
y = df_model["is_delayed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   ✓ Model Accuracy : {accuracy*100:.1f}%")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=["On Time", "Delayed"]))

# Feature importance chart
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
fig2, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax, color="#2E6DA4")
ax.set_title("Feature Importance — Random Forest", fontweight="bold", color="#1B2A4A")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("/home/claude/airline_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ Feature importance chart saved")

# ── STEP 6: Export Results ────────────────────────────────────────────────────
print("\n[6/6] Exporting results to Excel...")
summary = df.groupby(["route", "carrier", "season"]).agg(
    total_flights   = ("is_delayed", "count"),
    delayed_flights = ("is_delayed", "sum"),
    avg_delay_min   = ("delay_minutes", "mean"),
    avg_satisfaction= ("satisfaction", "mean")
).reset_index()
summary["delay_rate_%"] = (summary["delayed_flights"] / summary["total_flights"] * 100).round(1)
summary["avg_delay_min"] = summary["avg_delay_min"].round(1)
summary["avg_satisfaction"] = summary["avg_satisfaction"].round(2)
summary.to_excel("/home/claude/airline_delay_summary.xlsx", index=False)
print("   ✓ Summary exported: airline_delay_summary.xlsx")

print("\n" + "=" * 60)
print("  ANALYSIS COMPLETE")
print(f"  Model Accuracy  : {accuracy*100:.1f}%")
print(f"  Total Records   : {len(df):,}")
print(f"  Delayed Flights : {df['is_delayed'].sum():,} ({df['is_delayed'].mean()*100:.1f}%)")
print("=" * 60)
