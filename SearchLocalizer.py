import pandas as pd

series_id = "1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647"
localizer_df = pd.read_csv("train_localizers.csv")

# Filter to this SeriesInstanceUID
matches = localizer_df[localizer_df["SeriesInstanceUID"] == series_id]
print(f"Found {len(matches)} annotations for this series.")
print(matches.head())
