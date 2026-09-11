from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# ================================================================
# 1. PATHS SETUP & ARTIFACT LOADING
# ================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUTPUT_DIR / "feature_engineered_campaign_data.csv")
reg_model = joblib.load(MODEL_DIR / "revenue_regression_model.pkl")
reg_cols = joblib.load(MODEL_DIR / "regression_features.pkl")
clf_model = joblib.load(MODEL_DIR / "profit_classification_model.pkl")
clf_cols = joblib.load(MODEL_DIR / "classification_features.pkl")

# ================================================================
# 2. EXTRACT REVENUE DRIVERS (REGRESSION - RANDOM FOREST)
# ================================================================
importances = pd.Series(reg_model.feature_importances_, index=reg_cols).sort_values(ascending=False)
top_reg_drivers = importances.head(5).to_frame(name="Importance Score")

# ================================================================
# 3. EXTRACT PROFITABILITY DRIVERS (MODEL-AGNOSTIC CLASSIFIER)
# ================================================================
if hasattr(clf_model, "coef_"):
    # Logistic Regression path
    clf_weights = pd.Series(clf_model.coef_[0], index=clf_cols).sort_values(ascending=False)
    top_pos_clf = clf_weights.head(4).to_frame(name="Positive Weight (Drives Profit)")
    top_neg_clf = clf_weights.tail(4).to_frame(name="Negative Weight (Drives Loss Risk)")
    
    clf_summary = (
        "#### Features Positively Driving Profitability:\n\n"
        f"{top_pos_clf.to_markdown()}\n\n"
        "#### Features Increasing Loss Risk:\n\n"
        f"{top_neg_clf.to_markdown()}"
    )
elif hasattr(clf_model, "feature_importances_"):
    # Tree-based model path (Decision Tree / Random Forest)
    clf_importances = pd.Series(clf_model.feature_importances_, index=clf_cols).sort_values(ascending=False)
    top_clf_drivers = clf_importances.head(5).to_frame(name="Gini Importance Score")
    
    clf_summary = (
        "#### Top Determinants of Profit vs. Loss (Tree Importance):\n\n"
        f"{top_clf_drivers.to_markdown()}"
    )
else:
    clf_summary = "Model parameters are not directly inspectable for feature importance."

# ================================================================
# 4. COMPOSE & SAVE EXECUTIVE MARKETING REPORT
# ================================================================
report_content = f"""# Executive Marketing Intelligence Report
## Omnichannel Campaign Performance: Nykaa, Purplle, & Tira

---

### 1. Key Drivers of Revenue (Random Forest Regressor, R² ≈ 0.9625)
The regression model isolates the following top 5 features driving the highest variance in total gross revenue:

{top_reg_drivers.to_markdown()}

---

### 2. Profitability Determinants (Classification Analysis)
{clf_summary}

---

### 3. Actionable Strategic Recommendations

1. **Acquisition Efficiency Thresholds:**
   * Enforce a strict ceiling on `Cost_Per_Conv` relative to customer average order values (~₹550). When customer acquisition costs exceed this boundary, campaigns systematically shift from profitable to net loss.

2. **Channel Synergy & Allocation:**
   * Allocate primary spend toward multi-channel mixes that balance discovery (Instagram/YouTube) with direct conversion intent (Google Search). Avoid high-spend standalone campaigns that lack search capture support.

3. **Duration Optimization & Ad Fatigue:**
   * Monitor performance decays past standard duration windows. Campaigns with longer active runtimes require creative refreshes to avoid CTR decline and lead conversion rate erosion.

4. **Pre-Campaign Simulation (Decision Support):**
   * Before committing operational media budget, run target audience, channel mix, duration, and planned spend scenarios through the interactive dashboard (`app.py`) to review forecasted revenue and profit risk flags.
"""

report_path = OUTPUT_DIR / "Executive_Marketing_Report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print("=" * 70)
print(f"Executive Report successfully generated at:")
print(f"-> {report_path}")
print("=" * 70)