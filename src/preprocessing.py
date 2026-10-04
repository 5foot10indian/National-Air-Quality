import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

import os
import sys

# Add the main project folder (d:\Projects\Pollution) to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# NOW you can import safely (Note: do NOT include '.py' in the import)
from src.config import pollutants


print(pollutants)  # Just to check if it worked!



df = pd.read_csv("data/raw/city_day.csv")
df["Date"] = pd.to_datetime(df["Date"])
#print(df.head())

df.loc[df["PM2.5"]<0, "PM2.5"] = np.nan
df.drop_duplicates(subset=["City", "Date"])

print(df["PM10"].isnull().mean())  # overall % missing
print(df.groupby("City")["PM10"].apply(lambda x: x.isnull().mean()).sort_values(ascending=False))
df = df.drop(columns=["PM10"])

print(df["Xylene"].isnull().mean())
print(df.groupby("City")["Xylene"].apply(lambda x: x.isnull().mean()).sort_values(ascending=False))
df = df.drop(columns=["Xylene"])

imputer = SimpleImputer(strategy="median")
df[["PM2.5"]] = imputer.fit_transform(df[["PM2.5",]])

from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)
df[pollutants] = knn_imputer.fit_transform(df[pollutants])

#LOOKED FOR NULL VALUES BEFORE DELETING THE COLUMN BECAUSE OF HGIHER NUMBER OF NULL VALUES
#print(df["Xylene"].isna().mean()*100)


df["PM2.5_log"] = np.log1p(df["PM2.5"])

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="yeo-johnson")
df[["PM2.5_transformed"]] = pt.fit_transform(df[["PM2.5"]])

from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#df_scaled = scaler.fit_transform(df[pollutants])

from sklearn.preprocessing import RobustScaler
#r_scaler = RobustScaler()
#df_scaled = r_scaler.fit_transform(df[pollutants])

q1 = df["PM2.5"].quantile(0.25)
q3 = df["PM2.5"].quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr

df["PM2.5_capped"] = np.where(df["PM2.5"] > upper_limit, upper_limit, df["PM2.5"])

#Feature Engineering
df["Month"] = df["Date"].dt.month
def get_season(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in [3,4,5]:
        return "Summer"
    elif month in [6,7,8,9]:
        return "Monsoon"    
    else:
        return "Post-Monsoon"

df["Season"] = df["Month"].apply(get_season)   


df["Is_Diwali_Season"] = df["Date"].between("2019-10-20", "2019-10-30").astype(int)

city_features = df.groupby("City")["PM2.5"].agg(
    avg_day = "mean",
    volatality = "std",
    worst_day = "max"
).reset_index()

corr = df[pollutants].corr()
high_corr = corr[(corr > 0.85) & (corr < 1.0)]
print(high_corr.dropna(how="all"))

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_pipeline = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, pollutants)
])
X_processed = preprocessor.fit_transform(df)

df.to_csv("data/processed_pollution_data.csv", index=False)

print("Preprocessing complete! Saved to data/processed_pollution_data.csv")