from pathlib import Path
import pandas as pd

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load datasets
tira_df = pd.read_csv(DATA_DIR / "tira_campaign_data_with_nulls.csv")
nykaa_df = pd.read_csv(DATA_DIR / "nykaa_campaign_data_with_nulls.csv")
purplle_df = pd.read_csv(DATA_DIR / "purplle_campaign_data_with_nulls.csv")

# Add Brand_Name column
tira_df.insert(1, "Brand_Name", "Tira")
nykaa_df.insert(1, "Brand_Name", "Nykaa")
purplle_df.insert(1, "Brand_Name", "Purplle")

# Display results
datasets = {
    "Tira": tira_df,
    "Nykaa": nykaa_df,
    "Purplle": purplle_df
}

for brand, df in datasets.items():

    print("\n" + "=" * 70)
    print(f"{brand.upper()} DATASET AFTER ADDING BRAND")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nBrand Name Count:")
    print(df["Brand_Name"].value_counts())

    print("\nFirst 5 Records:")
    print(df.head())

# Combine all three datasets
combined_df = pd.concat(
    [tira_df, nykaa_df, purplle_df],
    ignore_index=True
)

print("\n" + "=" * 70)
print("COMBINED DATASET")
print("=" * 70)

print("\nShape:")
print(combined_df.shape)

print("\nBrand Distribution:")
print(combined_df["Brand_Name"].value_counts())

print("\nColumns:")
print(combined_df.columns.tolist())