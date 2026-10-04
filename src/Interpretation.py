import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/anomaly_detected_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

pollutants = ["PM2.5", "NO", "NO2", "NOx", "NH3",
              "CO", "SO2", "O3", "Benzene", "Toluene"]



profile = df.groupby("Cluster")[pollutants].mean()

overall_mean = df[pollutants].mean()
profile_relative = profile / overall_mean

print(profile_relative.round(2))

city_composition = pd.crosstab(df["Cluster"], df["City"], normalize="index")
top_cities_per_cluster = city_composition.apply(lambda row: row.sort_values(ascending=False).head(5), axis=1)

season_composition = pd.crosstab(df["Cluster"], df["Season"], normalize="index")

print(season_composition.round(2))

plt.figure(figsize=(10, 6))
sns.heatmap(profile_relative, annot=True, fmt=".2f", cmap="coolwarm", center=1)
plt.title("Cluster Pollutant Profiles (relative to overall average)")
plt.xlabel("Pollutant")
plt.ylabel("Cluster")
plt.show()

cluster_names = {
    0: "Moderate Urban Baseline",
    1: "Severe Winter Smog Belt",
    2: "Clean / Coastal Low-Pollution",
    3: "Traffic-Dominant, Moderate Particulates"
}



df["Cluster_Name"] = df["Cluster"].map(cluster_names)
if "AQI_Bucket" in df.columns:
    validation = pd.crosstab(df["Cluster_Name"], df["AQI_Bucket"], normalize="index")
    print(validation.round(2))



# If AQI_Bucket wasn't carried through, merge it back in:
raw = pd.read_csv("data/raw/city_day.csv")[["City", "Date", "AQI_Bucket"]]
raw["Date"] = pd.to_datetime(raw["Date"])
df = df.merge(raw, on=["City", "Date"], how="left")



summary = profile_relative.copy()
summary["Dominant_Season"] = season_composition.idxmax(axis=1)
summary["Top_City"] = city_composition.idxmax(axis=1)
summary["Cluster_Name"] = summary.index.map(cluster_names)

summary.to_csv("data/processed/cluster_summary.csv")
df.to_csv("data/processed/final_dataset.csv", index=False)

print(summary)