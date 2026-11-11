
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load saved model and feature columns
model = joblib.load("loan_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Load original dataset
df = pd.read_csv("Dataset.csv")

# Handle missing values (same as training)
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
    df["Loan_Amount_Term"].mean()
)
df["Credit_History"] = df["Credit_History"].fillna(
    df["Credit_History"].mean()
)

for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# Feature engineering
df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

df["ApplicantIncomelog"] = np.log1p(df["ApplicantIncome"])
df["LoanAmountlog"] = np.log1p(df["LoanAmount"])
df["Loan_Amount_Term_log"] = np.log1p(df["Loan_Amount_Term"])
df["Total_Income_log"] = np.log1p(df["Total_Income"])

# Encode categorical columns (same as training)
encoding = {
    "Gender": {"Female": 0, "Male": 1},
    "Married": {"No": 0, "Yes": 1},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}
}

for col, mapping in encoding.items():
    df[col] = df[col].astype(str).map(mapping)

# Remove rows with invalid encoded values, if any
df = df.dropna(subset=list(encoding.keys()))

# Select exactly the features used by the saved model
X = df[feature_columns]

print("Dataset prepared:", X.shape)
print("Features:", list(X.columns))

# Select background data
background = X.sample(
    n=min(100, len(X)),
    random_state=42
)

# Create SHAP TreeExplainer
explainer = shap.TreeExplainer(
    model,
    data=background
)

# Explain a sample of predictions
X_sample = X.sample(
    n=min(50, len(X)),
    random_state=42
)

shap_explanation = explainer(X_sample)

# Select the approved class (class 1)
values = shap_explanation.values
base_values = shap_explanation.base_values

if values.ndim == 3:
    values = values[:, :, 1]

if np.ndim(base_values) == 2:
    base_values = base_values[:, 1]

# Create a SHAP Explanation object
explanation = shap.Explanation(
    values=values,
    base_values=base_values,
    data=X_sample.values,
    feature_names=feature_columns
)

# Global feature importance
plt.figure()
shap.plots.bar(explanation, show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shap_feature_importance.png")
plt.show()

# SHAP summary plot
plt.figure()
shap.plots.beeswarm(explanation, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.show()

# Individual prediction explanation
plt.figure()
shap.plots.waterfall(explanation[0], show=False)
plt.title("Individual Loan Prediction Explanation")
plt.tight_layout()
plt.savefig("shap_waterfall.png")
plt.show()

print("\nSHAP analysis completed successfully!")
print("Generated:")
print("1. shap_feature_importance.png")
print("2. shap_summary.png")
print("3. shap_waterfall.png")
