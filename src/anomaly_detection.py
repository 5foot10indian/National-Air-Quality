import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

df = pd.read_csv("data\processed\clustering_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
from src.config import pollutants

iso_forest = IsolationForest(contamination=0.02, random_state=42)
df["Anomaly_IF"] = iso_forest.fit_predict(df[pollutants])
df["Anomaly_Score_IF"] = iso_forest.decision_function(df[pollutants])

print(df["Anomaly_IF"].value_counts())

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
df["Anomaly_LOF"] = lof.fit_predict(df[pollutants])
df["Anomaly_Score_LOF"] = lof.negative_outlier_factor_

print(df["Anomaly_LOF"].value_counts())

both_flagged = df[(df["Anomaly_IF"] == -1) & (df["Anomaly_LOF"] == -1)]
print(f"Flagged by both methods: {len(both_flagged)} rows")

print(both_flagged[["City", "Date"] + pollutants].sort_values("PM2.5", ascending=False).head(20))

delhi = df[df["City"] == "Delhi"].sort_values("Date")

plt.figure(figsize=(14, 5))
plt.plot(delhi["Date"], delhi["PM2.5"], alpha=0.5, label="PM2.5")

anomalies = delhi[delhi["Anomaly_IF"] == -1]
plt.scatter(anomalies["Date"], anomalies["PM2.5"], color="red", label="Anomaly", zorder=5)

plt.legend()
plt.title("Delhi PM2.5 with Flagged Anomalies")
#plt.show()
    
df.to_csv("data/processed/anomaly_detected_data.csv", index=False)