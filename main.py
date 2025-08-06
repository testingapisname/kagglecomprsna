import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV (adjust filename if needed)
df = pd.read_csv("train.csv")

# Clean column names
df.columns = df.columns.str.strip()

# 1. Count of 'Aneurysm Present'
aneurysm_counts = df['Aneurysm Present'].value_counts()

# 2. Most common aneurysm locations
location_columns = [
    col for col in df.columns 
    if col not in ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'Aneurysm Present']
]

location_sums = df[location_columns].sum().sort_values(ascending=False)

# 3. Correlation between aneurysm locations
location_corr = df[location_columns].corr()

# 4. Count of studies with multiple aneurysms
df['aneurysm_count'] = df[location_columns].sum(axis=1)
multi_aneurysm_count = (df['aneurysm_count'] > 1).sum()
total_cases = len(df)

# ==== PLOTS ====
# Plot 1: Aneurysm Present distribution
plt.figure(figsize=(6, 4))
aneurysm_counts.plot(kind='bar', title='Aneurysm Present')
plt.xticks(rotation=0)
plt.ylabel("Number of Series")
plt.xlabel("Aneurysm Present (0=No, 1=Yes)")
plt.show()

# Plot 2: Aneurysm location frequency
plt.figure(figsize=(12, 5))
location_sums.plot(kind='bar', title='Aneurysm Location Frequencies')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Plot 3: Location correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(location_corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Between Aneurysm Locations")
plt.show()

# Text Summary
print(f"\nStudies with multiple aneurysm locations: {multi_aneurysm_count} / {total_cases}")
