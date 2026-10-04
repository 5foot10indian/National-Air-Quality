# Discovering Air Pollution Patterns Across Indian Cities 🇮🇳

An unsupervised machine learning project that analyzes 5 years (2015–2020) of Central Pollution Control Board (CPCB) air quality data to discover natural pollution "personalities" among Indian cities, detect genuinely anomalous pollution events (Diwali spikes, COVID lockdown drops), and validate those findings against real-world context — all without using any pre-existing labels.

## Why this project

India's air pollution crisis is highly seasonal, regional, and event-driven (winter smog, Diwali firecrackers, crop-residue burning, monsoon washout). This makes it an ideal real-world dataset for learning the full unsupervised ML lifecycle: messy, correlated, partially missing, and genuinely interesting to interpret.

## What this project does

1. **Exploratory Data Analysis** — understands the shape, gaps, distributions, and seasonal patterns in the raw data
2. **Preprocessing** — cleans, imputes, transforms, and scales 11 pollutant measurements across ~30,000 city-day records
3. **Dimensionality Reduction (PCA)** — compresses highly correlated pollutants into 5 uncorrelated components
4. **Clustering (KMeans)** — groups city-days into distinct pollution "personalities," validated via elbow method and silhouette score
5. **Anomaly Detection (Isolation Forest + Local Outlier Factor)** — flags statistically unusual days, cross-validated by two independent methods, and matched against known real-world events
6. **End-to-end Pipeline** — packages preprocessing + PCA into a single reusable, saveable `scikit-learn` pipeline for consistent use on future data
7. **Interpretation** — translates cluster numbers into named, human-readable pollution profiles, validated against the dataset's official AQI bucket labels (held out of training)

## Key findings

*(Fill this in once you finalize your results — example format below)*

- Identified **4 distinct pollution clusters** among Indian cities, ranging from a "Severe Winter Smog Belt" (dominated by Delhi, Ghaziabad, Patna) to a "Clean/Coastal" group (Kochi, Shillong)
- Flagged **591 anomalous city-days** (~2%), with high-confidence anomalies (flagged by both detection methods) clustering strongly around **Diwali week** and the **March–May 2020 lockdown period**
- Cluster membership showed strong agreement with the dataset's official AQI severity labels, despite those labels never being used during training

## Project structure

```
air-quality-clustering/
├── data/
│   ├── raw/                  # original CPCB city_day.csv
│   └── processed/            # outputs from each pipeline stage
├── notebooks/                # exploratory analysis (optional, if used)
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── pca.py
│   ├── clustering.py
│   ├── anomaly_detection.py
│   ├── build_pipeline.py
│   └── interpretation.py
├── models/                   # saved .pkl pipeline and model files
├── app.py                    # optional Streamlit dashboard
├── requirements.txt
└── README.md
```

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `scipy` · `joblib` · `streamlit` (optional dashboard)

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/air-quality-clustering.git
cd air-quality-clustering

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline stages in order
python src/preprocessing.py
python src/pca.py
python src/clustering.py
python src/anomaly_detection.py
python src/build_pipeline.py
python src/interpretation.py

# 4. (Optional) Launch the dashboard
streamlit run app.py
```

## Dataset

[Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) — CPCB station-level and city-level daily pollutant readings, sourced via Kaggle / data.gov.in.

## What I learned

- How to diagnose and treat different *types* of missing data (random vs. structural)
- Why scaling and transformation order matters before feeding data into distance-based algorithms
- How to choose between KMeans, Hierarchical, DBSCAN, and GMM based on data shape assumptions
- How to validate unsupervised results against held-out ground truth without ever training on it
- How to build a production-style, reusable `scikit-learn` pipeline rather than ad-hoc scripts

## Possible extensions

- Correlate pollution clusters with public health data (hospital admissions, respiratory illness rates)
- Overlay NASA FIRMS satellite stubble-burning data to connect clusters to likely causes
- Build a predictive model (supervised) to forecast tomorrow's pollution cluster
- Map clusters geographically using `geopandas`
- Track individual cities' cluster membership over time to assess improvement/decline

## Author

*(Your name, LinkedIn/portfolio link)*
