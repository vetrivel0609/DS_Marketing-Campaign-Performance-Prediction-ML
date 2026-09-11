from pathlib import Path
import pandas as pd

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load the three datasets
tira_df = pd.read_csv(DATA_DIR / "tira_campaign_data_with_nulls.csv")
nykaa_df = pd.read_csv(DATA_DIR / "nykaa_campaign_data_with_nulls.csv")
purplle_df = pd.read_csv(DATA_DIR / "purplle_campaign_data_with_nulls.csv")

# Store datasets
datasets = {
    "Tira": tira_df,
    "Nykaa": nykaa_df,
    "Purplle": purplle_df
}

# Initial inspection
for brand, df in datasets.items():

    print("\n" + "=" * 70)
    print(f"{brand.upper()} CAMPAIGN DATASET")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())