import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

df = pd.read_csv("city_day.csv")
#print(df.shape)
#print(df.head())
#df.info()
df["Date"] = pd.to_datetime(df["Date"])
pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
              "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
okay = df[pollutants].describe()

missing_pct = df.isnull().mean() * 100
#print(missing_pct.sort_values(ascending=False))
missing_by_city = df.groupby("City")[pollutants].apply(lambda x: x.isnull().mean() * 100)


#plt.figure(figsize=(12, 8))
#sns.heatmap(missing_by_city, annot=True, fmt=".0f", cmap="Reds")
#plt.show()

col = "PM2.5"
#print("Mean:", df[col].mean())
#print("Median:", df[col].median())
#print("Skew:", df[col].skew())

#fig, axes = plt.subplots(1, 2, figsize=(12, 4))    # 1 row, 2 plots side by side
#sns.histplot(df[col], kde=True, ax=axes[0])
#sns.boxplot(x=df[col], ax=axes[1])
#plt.show()

# Skewness of all pollutants at once
df[pollutants].skew().sort_values(ascending=False)
corr = df[pollutants].corr()

#plt.figure(figsize=(10, 8))
#sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
#plt.show()

#sns.scatterplot(x="PM2.5", y="PM10", data=df, alpha=0.3)
#plt.show()

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

#df.groupby("Month")["PM2.5"].mean().plot(kind="bar", figsize=(10,4))
#plt.ylabel("Average PM2.5")
#plt.show()

#delhi = df[df["City"] == "Delhi"].set_index("Date")
#delhi["PM2.5"].plot(alpha=0.4, figsize=(12,4), label="Daily")
#delhi["PM2.5"].rolling(30).mean().plot(label="30-day Average")
#plt.legend()
#plt.show()

city_summary = df.groupby("City")["PM2.5"].agg(["mean", "median","std","max"])
#print(city_summary.sort_values("mean", ascending=False))
pivot = df.pivot_table(values="PM2.5", index="City", columns="Month", aggfunc="mean")

#plt.figure(figsize=(12, 8))
#sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f")
#plt.show()

#print("Duplicate rows:", df.duplicated().sum())
#print("Same city+date twice:", df.duplicated(subset=["City", "Date"]).sum())

for col in pollutants:
    n_negative = (df[col]<0).sum()
    #print(col,"negative values:", n_negative)

aeooaa = df.groupby("City")["Date"].agg(["min", "max", "count"])
#print(aeooaa)



 