# ✈️ Airline Delay & Customer Satisfaction Analysis

## Project Overview
End-to-end data analysis and machine learning project on airline operations data.
Identifies root causes of flight delays and predicts delay likelihood using a
Random Forest classifier — achieving **95.9% accuracy** on 50,000 records.

## Tools & Technologies
- **Python** — Pandas, NumPy, Scikit-learn, Matplotlib
- **Machine Learning** — Random Forest Classifier
- **Output** — Charts (PNG) + Excel Summary Report

## Project Structure
```
project1_airline_delay/
│
├── airline_delay_analysis.py   ← Main analysis script
├── airline_eda_charts.png      ← EDA visualisations (auto-generated)
├── airline_feature_importance.png ← ML feature importance (auto-generated)
├── airline_delay_summary.xlsx  ← Summary report (auto-generated)
└── README.md
```

## How to Run
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl

# Run the analysis
python airline_delay_analysis.py
```

## Key Findings
- **Monsoon season** has the highest average delay (54.1 min)
- **BOM-BLR** route is the most delay-prone
- **Weather** is the #1 cause of delays (20% of flights)
- Random Forest model achieves **95.9% prediction accuracy**
- Customer satisfaction drops sharply when delay exceeds 40 minutes

## What the Script Does
1. Generates a realistic 50,000-row airline operations dataset
2. Cleans and validates all records
3. Performs EDA — delay by carrier, season, route, and cause
4. Visualises findings in a 4-panel dashboard chart
5. Trains a Random Forest model to predict flight delays
6. Exports full summary to Excel for stakeholder reporting

## Author
**Arun Kumar** | [LinkedIn](https://linkedin.com/in/arun-kumar-a45685225) | [GitHub](https://github.com/arun70049)
