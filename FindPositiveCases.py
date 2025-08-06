import pandas as pd

df = pd.read_csv("train_localizers.csv")
positive_series_ids = df["SeriesInstanceUID"].unique()
print(f"Total series with aneurysm annotations: {len(positive_series_ids)}")

# Print a few SeriesInstanceUIDs to choose from
print("Example series with annotations:")
for uid in positive_series_ids[:5]:
    print(uid)
