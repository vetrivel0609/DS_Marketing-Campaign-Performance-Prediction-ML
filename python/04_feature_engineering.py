from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD CLEANED DATA
# ============================================================

input_file = OUTPUT_DIR / "cleaned_campaign_data.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("LOADING CLEANED DATA")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. CREATE PROFIT FLAG
# ============================================================

print("\n" + "=" * 70)
print("CREATING PROFIT FLAG")
print("=" * 70)

# ROI > 0  → Profit
# ROI <= 0 → Loss / Break-even

df["Profit_Flag"] = np.where(
    df["ROI"] > 0,
    1,
    0
)

print("\nProfit_Flag created successfully.")

print("\nProfit_Flag distribution:")
print(df["Profit_Flag"].value_counts())

print("\nProfit_Flag percentage:")
print(
    df["Profit_Flag"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 4. CHECK CHANNEL VALUES
# ============================================================

print("\n" + "=" * 70)
print("CHANNEL ANALYSIS")
print("=" * 70)

print("\nNumber of unique Channel_Used values:")
print(df["Channel_Used"].nunique())

print("\nChannel values:")
print(df["Channel_Used"].unique())


# ============================================================
# 5. CLEAN CHANNEL TEXT
# ============================================================

df["Channel_Used"] = (
    df["Channel_Used"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 6. MULTI-LABEL ENCODING
# ============================================================

print("\n" + "=" * 70)
print("MULTI-LABEL ENCODING OF CHANNEL_USED")
print("=" * 70)

# Actual individual marketing channels present in the dataset
channels = [
    "Email",
    "Facebook",
    "Google",
    "Instagram",
    "WhatsApp",
    "YouTube"
]

# Create one binary column for each channel
# 1 = channel is used
# 0 = channel is not used

for channel in channels:

    df[channel] = (
        df["Channel_Used"]
        .str.contains(
            channel,
            regex=False,
            na=False
        )
        .astype(int)
    )

print("\nEncoded Channel Columns:")
print(channels)

print("\nNumber of encoded channel columns:")
print(len(channels))


# ============================================================
# 7. FEATURE-ENGINEERED DATASET CHECK
# ============================================================

print("\n" + "=" * 70)
print("FEATURE-ENGINEERED DATASET")
print("=" * 70)

print("\nFeature-engineered dataset shape:")
print(df.shape)

print("\nEncoded Channel Sample:")
print(
    df[
        [
            "Channel_Used",
            "Email",
            "Facebook",
            "Google",
            "Instagram",
            "WhatsApp",
            "YouTube"
        ]
    ].head(10)
)


# ============================================================
# 8. CHECK PROFIT FLAG
# ============================================================

print("\n" + "=" * 70)
print("PROFIT FLAG VALIDATION")
print("=" * 70)

validation = df.groupby("Profit_Flag")["ROI"].agg(
    Count="count",
    Minimum_ROI="min",
    Maximum_ROI="max"
)

print(validation)


# ============================================================
# 9. CHECK MULTI-LABEL ENCODING
# ============================================================

print("\n" + "=" * 70)
print("MULTI-LABEL ENCODING VALIDATION")
print("=" * 70)

print("\nOriginal Channel_Used:")
print(
    df["Channel_Used"]
    .head(10)
    .to_string(index=False)
)

print("\nEncoded Channel Columns:")

print(
    df[
        [
            "Email",
            "Facebook",
            "Google",
            "Instagram",
            "WhatsApp",
            "YouTube"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 10. CHECK FOR DUPLICATE COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("COLUMN VALIDATION")
print("=" * 70)

duplicate_columns = df.columns[
    df.columns.duplicated()
].tolist()

print("\nDuplicate Column Names:")

if duplicate_columns:
    print(duplicate_columns)
else:
    print("None")


# ============================================================
# 11. FINAL DATASET CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL FEATURE-ENGINEERED DATASET")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

# Check for duplicate column names
assert not df.columns.duplicated().any(), \
    "Duplicate column names detected!"

# Check missing values
assert df.isnull().sum().sum() == 0, \
    "Missing values detected!"

# Check Profit_Flag values
assert set(df["Profit_Flag"].unique()).issubset({0, 1}), \
    "Invalid Profit_Flag values detected!"

# Check encoded channel values
for channel in channels:
    assert set(df[channel].unique()).issubset({0, 1}), \
        f"Invalid values detected in {channel} column!"

print("\nAll validation checks passed successfully!")


# ============================================================
# 13. SAVE FEATURE-ENGINEERED DATASET
# ============================================================

output_file = OUTPUT_DIR / "feature_engineered_campaign_data.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("FILE SAVED")
print("=" * 70)

print("\nFeature-engineered dataset saved to:")
print(output_file)

print("\nFeature Engineering completed successfully!")