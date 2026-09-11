from pathlib import Path
import pandas as pd
import numpy as np


# ================================================================
# PROJECT DIRECTORIES
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create outputs folder if it does not exist
OUTPUT_DIR.mkdir(exist_ok=True)


# ================================================================
# LOAD RAW DATASETS
# ================================================================

tira_df = pd.read_csv(
    DATA_DIR / "tira_campaign_data_with_nulls.csv"
)

nykaa_df = pd.read_csv(
    DATA_DIR / "nykaa_campaign_data_with_nulls.csv"
)

purplle_df = pd.read_csv(
    DATA_DIR / "purplle_campaign_data_with_nulls.csv"
)


# ================================================================
# ADD BRAND NAME
# ================================================================

tira_df["Brand_Name"] = "Tira"
nykaa_df["Brand_Name"] = "Nykaa"
purplle_df["Brand_Name"] = "Purplle"


# ================================================================
# COMBINE ALL THREE DATASETS
# ================================================================

df = pd.concat(
    [tira_df, nykaa_df, purplle_df],
    ignore_index=True
)


# ================================================================
# MOVE BRAND_NAME AFTER CAMPAIGN_ID
# ================================================================

columns = df.columns.tolist()

columns.remove("Brand_Name")

campaign_id_position = columns.index("Campaign_ID")

columns.insert(
    campaign_id_position + 1,
    "Brand_Name"
)

df = df[columns]


# ================================================================
# INITIAL DATASET INFORMATION
# ================================================================

print("=" * 70)
print("INITIAL COMBINED DATASET")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ================================================================
# 1. REMOVE DUPLICATE RECORDS
# ================================================================

duplicates_before = df.duplicated().sum()

df = df.drop_duplicates()

print("\n" + "=" * 70)
print("DUPLICATE RECORD HANDLING")
print("=" * 70)

print("\nDuplicate records found:", duplicates_before)
print("Shape after removing duplicates:", df.shape)


# ================================================================
# 2. REMOVE ROWS WITH MISSING CAMPAIGN_ID
# ================================================================

missing_id_before = df["Campaign_ID"].isna().sum()

df = df.dropna(subset=["Campaign_ID"])

print("\n" + "=" * 70)
print("CAMPAIGN ID HANDLING")
print("=" * 70)

print("\nRows with missing Campaign_ID:", missing_id_before)
print("Rows removed:", missing_id_before)
print("Shape after removing missing Campaign_ID:", df.shape)


# ================================================================
# 3. STANDARDIZE CAMPAIGN_ID
# ================================================================

df["Campaign_ID"] = (
    df["Campaign_ID"]
    .astype(str)
    .str.strip()
)


# ================================================================
# 4. STANDARDIZE BRAND_NAME
# ================================================================

df["Brand_Name"] = (
    df["Brand_Name"]
    .astype(str)
    .str.strip()
)


# ================================================================
# 5. CONVERT DATE TO DATETIME
# ================================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)


# ================================================================
# 6. CONVERT NUMERICAL COLUMNS
# ================================================================

numeric_columns = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "ROI",
    "Engagement_Score"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ================================================================
# 7. CHECK INVALID NEGATIVE VALUES
# ================================================================

non_negative_columns = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "Engagement_Score"
]

print("\n" + "=" * 70)
print("INVALID NEGATIVE VALUES")
print("=" * 70)

for column in non_negative_columns:

    count = (df[column] < 0).sum()

    print(f"{column}: {count}")

    # Replace invalid negative values with NaN
    df.loc[df[column] < 0, column] = np.nan


# ================================================================
# 8. HANDLE CATEGORICAL MISSING VALUES
# ================================================================

categorical_columns = [
    "Campaign_Type",
    "Target_Audience",
    "Channel_Used",
    "Language",
    "Customer_Segment"
]

print("\n" + "=" * 70)
print("CATEGORICAL MISSING VALUE HANDLING")
print("=" * 70)

for column in categorical_columns:

    missing_before = df[column].isna().sum()

    df[column] = df[column].fillna("Unknown")

    missing_after = df[column].isna().sum()

    print(
        f"{column}: "
        f"{missing_before} missing → "
        f"{missing_after} missing"
    )


# ================================================================
# 9. HANDLE NUMERICAL MISSING VALUES
# ================================================================
#
# Median imputation is performed separately for each brand.
#
# ROI is excluded because it will be recalculated later.
#


numeric_columns_without_roi = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "Engagement_Score"
]

print("\n" + "=" * 70)
print("NUMERICAL MISSING VALUE HANDLING")
print("=" * 70)

for column in numeric_columns_without_roi:

    missing_before = df[column].isna().sum()

    # Calculate median separately for each brand
    brand_median = df.groupby("Brand_Name")[column].transform("median")

    # Fill missing values with corresponding brand median
    df[column] = df[column].fillna(brand_median)

    missing_after = df[column].isna().sum()

    print(
        f"{column}: "
        f"{missing_before} missing → "
        f"{missing_after} missing"
    )


# ================================================================
# 10. HANDLE MISSING DATES
# ================================================================
#
# Missing dates are filled with the most frequent date
# within the corresponding brand.
#
# We use transform() instead of groupby().apply() so that
# Brand_Name remains in the dataset.
#


print("\n" + "=" * 70)
print("DATE MISSING VALUE HANDLING")
print("=" * 70)

missing_dates_before = df["Date"].isna().sum()

# Find the most frequent date for each brand
brand_date_mode = (
    df.groupby("Brand_Name")["Date"]
    .transform(
        lambda x: x.mode().iloc[0]
        if not x.mode().empty
        else pd.NaT
    )
)

# Fill missing dates using brand-specific mode
df["Date"] = df["Date"].fillna(brand_date_mode)

missing_dates_after = df["Date"].isna().sum()

print("\nMissing dates before:", missing_dates_before)
print("Missing dates after:", missing_dates_after)


# ================================================================
# 11. RECALCULATE ROI
# ================================================================
#
# IMPORTANT:
#
# The original ROI supplied in the raw datasets is incorrect.
#
# Therefore, the original ROI is completely replaced.
#
# Formula:
#
# ROI = (Revenue - Acquisition_Cost) / Acquisition_Cost
#


print("\n" + "=" * 70)
print("RECALCULATING ROI")
print("=" * 70)

# Check for zero acquisition cost
zero_acquisition_cost = (
    df["Acquisition_Cost"] == 0
).sum()

print("\nZero Acquisition_Cost values:", zero_acquisition_cost)


# Calculate ROI
df["ROI"] = (
    (df["Revenue"] - df["Acquisition_Cost"])
    / df["Acquisition_Cost"]
)


# Round ROI to 4 decimal places
df["ROI"] = df["ROI"].round(4)


# ================================================================
# 12. CHECK ROI FOR MISSING / INFINITE VALUES
# ================================================================

missing_roi = df["ROI"].isna().sum()
infinite_roi = np.isinf(df["ROI"]).sum()

print("\nMissing ROI:", missing_roi)
print("Infinite ROI:", infinite_roi)


# ================================================================
# 13. ROI VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("RECALCULATED ROI VALIDATION")
print("=" * 70)

print("\nROI Minimum:", df["ROI"].min())
print("ROI Maximum:", df["ROI"].max())
print("ROI Mean:", round(df["ROI"].mean(), 4))
print("ROI Median:", round(df["ROI"].median(), 4))

print("\nMissing ROI:", df["ROI"].isna().sum())
print("Infinite ROI:", np.isinf(df["ROI"]).sum())


# ================================================================
# 14. PROFIT / LOSS DISTRIBUTION
# ================================================================

profit_count = (
    df["ROI"] > 0
).sum()

loss_count = (
    df["ROI"] <= 0
).sum()


print("\n" + "=" * 70)
print("ROI PROFIT / LOSS DISTRIBUTION")
print("=" * 70)

print("\nProfitable campaigns (ROI > 0):", profit_count)
print("Loss / break-even campaigns (ROI <= 0):", loss_count)


# ================================================================
# 15. FINAL MISSING VALUE CHECK
# ================================================================

print("\n" + "=" * 70)
print("FINAL MISSING VALUE CHECK")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)


# ================================================================
# 16. FINAL DATASET INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("FINAL CLEANED DATASET")
print("=" * 70)

print("\nFinal Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nBrand Distribution:")
print(df["Brand_Name"].value_counts())

print("\nDuplicate Records:")
print(df.duplicated().sum())


# ================================================================
# 17. VERIFY ROI CALCULATION
# ================================================================

print("\n" + "=" * 70)
print("SAMPLE RECALCULATED ROI")
print("=" * 70)

sample_columns = [
    "Campaign_ID",
    "Brand_Name",
    "Revenue",
    "Acquisition_Cost",
    "ROI"
]

print(
    df[sample_columns]
    .head(10)
    .to_string(index=False)
)


# ================================================================
# 18. VERIFY FINAL DATASET
# ================================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print("\nNumber of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nBrand counts:")
print(df["Brand_Name"].value_counts())

print("\nTotal missing values:", df.isnull().sum().sum())

print("Total duplicate rows:", df.duplicated().sum())


# ================================================================
# 19. SAVE CLEANED DATASET
# ================================================================

output_file = OUTPUT_DIR / "cleaned_campaign_data.csv"

df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)
print("FILE SAVED")
print("=" * 70)

print("\nCleaned dataset saved to:")
print(output_file)

print("\nCleaning process completed successfully.")