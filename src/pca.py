import os
import sys
# Add the main project folder (d:\Projects\Pollution) to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# NOW you can import safely (Note: do NOT include '.py' in the import)



import pandas as pd
from src.config import pollutants
print(pollutants)  # Just to check if it worked!
df = pd.read_csv("data\processed_pollution_data.csv")
df["Date"] = pd.to_datetime(df["Date"])



from src.pipeline import this_pipeline
my_pipeline = this_pipeline()
X_processed = my_pipeline.fit_transform(df)




from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_processed)
# See what percentage of information each of the 5 components captured:
print(pca.explained_variance_ratio_)

# See the total percentage of information retained:
print(pca.explained_variance_ratio_.cumsum())
import matplotlib.pyplot as plt

#plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
        #pca.explained_variance_ratio_, marker="o")
#plt.xlabel("Component number")
#plt.ylabel("Explained variance")
#plt.title("Scree plot")
#plt.show()
#print(df.columns.tolist())
pca_full = PCA().fit(X_processed)
cum_var = pca_full.explained_variance_ratio_.cumsum()
n_components_needed = (cum_var < 0.90).sum() + 1
print(n_components_needed)

loadings = pd.DataFrame(pca.components_, columns=pollutants)
print(loadings.iloc[0].sort_values(ascending=False))

# Assuming you have:
# df            -> original dataframe with City, Date, and raw/cleaned columns
# X_processed   -> output of your preprocessing pipeline (numpy array)
# X_pca         -> output of PCA (numpy array)

# Assuming you have:
# df            -> original dataframe with City, Date, and raw/cleaned columns
# X_processed   -> output of your preprocessing pipeline (numpy array)
# X_pca         -> output of PCA (numpy array)

import pandas as pd

# 1. Turn the processed (pre-PCA) array back into a labelled DataFrame
processed_df = pd.DataFrame(X_processed, columns=pollutants, index=df.index)

# 2. Turn the PCA array into a labelled DataFrame
pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)

# 3. Stitch identifiers + processed features + PCA components together
final_df = pd.concat(
    [df[["City", "Date", "Season"]], processed_df, pca_df],
    axis=1
)

# 4. Export
final_df.to_csv("data/air_quality_preprocessed_pca.csv", index=False)

import joblib

from src.pipeline import this_pipeline

# 1. Get your full pipeline
my_pipeline = this_pipeline()

# 2. Fit it
X_processed = my_pipeline.fit_transform(df)

# 3. Save the whole pipeline instead
os.makedirs("models", exist_ok=True)
joblib.dump(my_pipeline, "models/pollution_pipeline.pkl")

print("Pipeline saved successfully!")
