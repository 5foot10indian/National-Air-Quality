import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import os
import sys
# Add the main project folder (d:\Projects\Pollution) to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

df = pd.read_csv("data/processed/air_quality_preprocessed_pca.csv")
df["Date"] = pd.to_datetime(df["Date"])
from src.config import pollutants

pca_cols = [c for c in df.columns if c.startswith("PC")]
X_pca = df[pca_cols].values

print("Using Columns for Clustering:", pca_cols)
print("Shape:", X_pca.shape)
inertia_list=[]
sil_scores=[]
k_range = range(2,11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    inertia_list.append(km.inertia_)
    sil_scores.append(silhouette_score(X_pca, labels))
"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(k_range, inertia_list, marker="o")
axes[0].set_title("Elbow Plot")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")

axes[1].plot(k_range, sil_scores, marker="o", color="green")
axes[1].set_title("Silhouette Scores")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Score")

#plt.tight_layout()
#plt.show()
"""


FINAL_K=4
final_km = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
df["Cluster"] = final_km.fit_predict(X_pca)
print(df["Cluster"].value_counts())

print(df.groupby("Cluster")[pollutants].mean())
print(df.groupby("Cluster")["City"].value_counts())
'''
sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=df, palette="tab10", alpha=0.6)
plt.title("Clusters in PCA space")
plt.show()
'''

df.to_csv("data/processed/clustering_data.csv", index=False)
print(df.head())
print(df["Cluster"].unique())