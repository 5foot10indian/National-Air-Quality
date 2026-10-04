import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA

df = pd.read_csv("data/raw/city_day.csv")
df["Date"] = pd.to_datetime(df["Date"])



pollutants = ["PM2.5", "NO", "NO2", "NOx", "NH3",
              "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
df["Month"] = df["Date"].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

df["Season"] = df["Month"].apply(get_season)
df = df.drop(columns=["PM10"])  # dropped due to excessive imputed duplication, found in Phase 5




numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("power_transform", PowerTransformer(method="yeo-johnson")),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, pollutants)
], remainder="drop")
full_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("pca", PCA(n_components=5))
])
X_pca = full_pipeline.fit_transform(df[pollutants])

pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)

final_df = pd.concat(
    [df[["City", "Date", "Season"]], pca_df],
    axis=1
)
final_df.to_csv("data/processed/pipeline_output.csv", index=False)

joblib.dump(full_pipeline, "models/full_preprocessing_pca_pipeline.pkl")

print("Pipeline fitted and saved.")
print(final_df.head())




'''
def this_pipeline():
    numeric_pipeline = Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, pollutants)
    ])

    return preprocessor
 '''