import pandas as pd
raw_df = pd.read_csv("data/raw/city_day.csv")
processed_df = pd.read_csv("data/processed_pollution_data.csv")
print(f"Raw file size: {raw_df.shape} (Rows, Columns)")
print(f"Processed file size: {processed_df.shape} (Rows, Columns)")
