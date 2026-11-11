import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION AND MODEL LOADING
# ============================================================

st.set_page_config(
    page_title="Intelligent Loan Approval Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns


try:
    model, feature_columns = load_artifacts()

except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.info(
        "Keep loan_model.pkl and feature_columns.pkl "
        "in the same folder as app.py."
    )
    st.stop()


# ============================================================
# 2. SHAP BACKGROUND DATA AND EXPLAINER
# ============================================================

@st.cache_resource
def load_shap_tools():

    dataset_path = Path("Dataset.csv")

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset.csv was not found. "
            "Place the original training dataset beside app.py."
        )

    df = pd.read_csv(dataset_path)

    required_columns = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing dataset columns: " + ", ".join(missing)
        )

    # Handle numerical missing values
    for col in [
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History"
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if col == "LoanAmount":
            fill_value = df[col].median()
        else:
            fill_value = df[col].mean()

        df[col] = df[col].fillna(fill_value)

    # Handle categorical missing values
    for col in [
        "Gender",
        "Married",
        "Dependents",
        "Self_Employed",
        "Education",
        "Property_Area"
    ]:

        df[col] = df[col].fillna(
            df[col].mode().iloc[0]
        )

    df["ApplicantIncome"] = pd.to_numeric(
        df["ApplicantIncome"],
        errors="coerce"
    )

    df["CoapplicantIncome"] = pd.to_numeric(
        df["CoapplicantIncome"],
        errors="coerce"
    )

    df["Total_Income"] = (
        df["ApplicantIncome"] +
        df["CoapplicantIncome"]
    )

    # Feature engineering
    df["ApplicantIncomelog"] = np.log1p(
        df["ApplicantIncome"].clip(lower=0)
    )

    df["LoanAmountlog"] = np.log1p(
        df["LoanAmount"].clip(lower=0)
    )

    df["Loan_Amount_Term_log"] = np.log1p(
        df["Loan_Amount_Term"].clip(lower=0)
    )

    df["Total_Income_log"] = np.log1p(
        df["Total_Income"].clip(lower=0)
    )

    # Encoding
    encoding_maps = {

        "Gender": {
            "Female": 0,
            "Male": 1
        },

        "Married": {
            "No": 0,
            "Yes": 1
        },

        "Education": {
            "Graduate": 0,
            "Not Graduate": 1
        },

        "Dependents": {
            "0": 0,
            "1": 1,
            "2": 2,
            "3+": 3
        },

        "Self_Employed": {
            "No": 0,
            "Yes": 1
        },

        "Property_Area": {
            "Rural": 0,
            "Semiurban": 1,
            "Urban": 2
        }

    }

    for col, mapping in encoding_maps.items():

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .map(mapping)
        )

    # Ensure model feature order
    for col in feature_columns:

        if col not in df.columns:
            raise ValueError(
                f"Feature {col} is missing from the dataset."
            )

    background = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    background = background.replace(
        [np.inf, -np.inf],
        np.nan
    )

    background = background.fillna(
        background.median()
    ).fillna(0)

    background = background.sample(
        n=min(100, len(background)),
        random_state=42
    )

    explainer = shap.TreeExplainer(
        model,
        data=background
    )

    return explainer, background


def shap_class_one_values(raw_values):

    if isinstance(raw_values, list):

        if len(raw_values) > 1:
            return np.asarray(raw_values[1])

        return np.asarray(raw_values[0])

    values = np.asarray(raw_values)

    if values.ndim == 3:

        if values.shape[-1] == 2:
            return values[:, :, 1]

        if values.shape[1] == 2:
            return values[:, 1, :]

    if values.ndim == 2:
        return values

    raise ValueError(
        f"Unsupported SHAP output shape: {values.shape}"
    )


def shap_class_one_base(explainer):

    base = np.asarray(
        explainer.expected_value
    )

    if base.ndim == 0:
        return float(base)

    if base.size > 1:
        return float(base.reshape(-1)[1])

    return float(base.reshape(-1)[0])


# ============================================================
# 3. DEFAULT VALUES AND PRESETS
# ============================================================

DEFAULTS = {

    "gender": "Male",
    "married": "Yes",
    "dependents": "0",
    "education": "Graduate",
    "self_employed": "No",
    "property_area": "Urban",

    "applicant_income": 5000,
    "coapplicant_income": 0,

    "loan_amount": 150,
    "loan_term": 360,

    "credit_history": "Good",
    "interest_rate": 9.0

}


PRESETS = {

    "Strong applicant": {

        "gender": "Male",
        "married": "Yes",
        "dependents": "1",
        "education": "Graduate",
        "self_employed": "No",
        "property_area": "Semiurban",

        "applicant_income": 8000,
        "coapplicant_income": 3000,

        "loan_amount": 140,
        "loan_term": 360,

        "credit_history": "Good"

    },

    "Risky applicant": {

        "gender": "Female",
        "married": "No",
        "dependents": "3+",
        "education": "Not Graduate",
        "self_employed": "Yes",
        "property_area": "Rural",

        "applicant_income": 2500,
        "coapplicant_income": 0,

        "loan_amount": 250,
        "loan_term": 120,

        "credit_history": "Bad"

    }

}


for key, value in DEFAULTS.items():

    st.session_state.setdefault(
        key,
        value
    )


st.session_state.setdefault("history", [])
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("counterfactual_results", None)


def apply_preset(name):

    for key, value in PRESETS[name].items():
        st.session_state[key] = value

    st.session_state["last_result"] = None
    st.session_state["counterfactual_results"] = None


def reset_form():

    for key, value in DEFAULTS.items():
        st.session_state[key] = value

    st.session_state["last_result"] = None
    st.session_state["counterfactual_results"] = None


# ============================================================
# 4. CSS DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        max-width: 1150px;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #1e3c72 0%,
            #2a5298 50%,
            #6a85b6 100%
        );
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(30,60,114,.25);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        color: white;
    }

    .hero p {
        margin: .4rem 0 0;
        opacity: .9;
        font-size: 1.05rem;
    }

    .result-approved {
        background: linear-gradient(
            135deg,
            #11998e,
            #38ef7d
        );
        color: white;
        padding: 1.6rem;
        border-radius: 16px;
        text-align: center;
    }

    .result-rejected {
        background: linear-gradient(
            135deg,
            #cb2d3e,
            #ef473a
        );
        color: white;
        padding: 1.6rem;
        border-radius: 16px;
        text-align: center;
    }

    .result-approved h2,
    .result-rejected h2 {
        margin: 0;
        color: white;
        font-size: 2rem;
    }

    .result-approved p,
    .result-rejected p {
        margin: .3rem 0 0;
        opacity: .95;
    }

    .meter-wrap {
        background: rgba(128,128,128,.2);
        border-radius: 20px;
        height: 18px;
        overflow: hidden;
    }

    .meter-fill {
        height: 100%;
        border-radius: 20px;
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.18);
        padding: .8rem 1rem;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def build_applicant(ss):

    gender_map = {
        "Female": 0,
        "Male": 1
    }

    married_map = {
        "No": 0,
        "Yes": 1
    }

    dependents_map = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3+": 3
    }

    education_map = {
        "Graduate": 0,
        "Not Graduate": 1
    }

    employment_map = {
        "No": 0,
        "Yes": 1
    }

    property_map = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }

    total_income = (
        ss["applicant_income"] +
        ss["coapplicant_income"]
    )

    applicant = {

        "Gender": gender_map[ss["gender"]],

        "Married": married_map[ss["married"]],

        "Dependents": dependents_map[ss["dependents"]],

        "Education": education_map[ss["education"]],

        "Self_Employed": employment_map[
            ss["self_employed"]
        ],

        "Credit_History": (
            1.0
            if ss["credit_history"] == "Good"
            else 0.0
        ),

        "Property_Area": property_map[
            ss["property_area"]
        ],

        "ApplicantIncomelog": np.log1p(
            ss["applicant_income"]
        ),

        "LoanAmountlog": np.log1p(
            ss["loan_amount"]
        ),

        "Loan_Amount_Term_log": np.log1p(
            ss["loan_term"]
        ),

        "Total_Income_log": np.log1p(
            total_income
        )

    }

    return applicant


def predict_applicant(applicant):

    input_df = pd.DataFrame(
        [applicant]
    )[feature_columns]

    prediction = int(
        model.predict(input_df)[0]
    )

    probability = None

    if hasattr(model, "predict_proba"):

        classes = list(model.classes_)

        if 1 in classes:

            probability = float(
                model.predict_proba(input_df)[0][
                    classes.index(1)
                ]
            )

    return input_df, prediction, probability


def calculate_emi(
    principal,
    months,
    annual_rate
):

    monthly_rate = annual_rate / 12 / 100

    if months <= 0:
        return 0.0

    if monthly_rate == 0:
        return principal / months

    factor = (
        1 + monthly_rate
    ) ** months

    return (
        principal
        * monthly_rate
        * factor
        / (factor - 1)
    )


# ============================================================
# 6. APPLICANT RISK CATEGORIZATION
# ============================================================

def categorize_applicant(probability):

    if probability is None:

        return (
            "Unavailable",
            "Prediction probability is unavailable."
        )

    percentage = probability * 100

    if percentage >= 70:

        return (
            "Low",
            "High predicted loan approval likelihood."
        )

    elif percentage >= 40:

        return (
            "Medium",
            "Moderate predicted loan approval likelihood."
        )

    else:

        return (
            "High",
            "Low predicted loan approval likelihood."
        )


# ============================================================
# 7. ELIGIBILITY IMPROVEMENT ADVISOR
# ============================================================

def generate_eligibility_advice(

    applicant_income,
    coapplicant_income,
    loan_amount_thousands,
    loan_term,
    credit_history,
    approval_probability,
    emi_ratio

):

    advice = []

    total_income = (
        applicant_income +
        coapplicant_income
    )

    annual_income = total_income * 12

    if credit_history == 0:

        advice.append(
            "Review your credit report, clear outstanding dues "
            "where possible, and maintain timely repayments."
        )

    if annual_income > 0:

        loan_to_income = (
            loan_amount_thousands * 1000
        ) / annual_income

        if loan_to_income > 3:

            advice.append(
                "The requested loan is relatively large compared "
                "with annual income. Consider whether a smaller "
                "loan would meet your needs."
            )

    if coapplicant_income == 0:

        advice.append(
            "If appropriate and eligible, a financially stable "
            "co-applicant may strengthen the household income profile."
        )

    if loan_term < 180:

        advice.append(
            "A longer repayment period may reduce monthly EMI, "
            "although it can increase total interest paid."
        )

    if emi_ratio is not None and emi_ratio > 40:

        advice.append(
            "The estimated EMI is high compared with monthly income. "
            "Consider a lower loan amount or a different repayment tenure."
        )

    if approval_probability is not None:

        if approval_probability < 0.40:

            advice.append(
                "The model indicates low approval likelihood. "
                "Review the financial factors before applying."
            )

        elif approval_probability < 0.70:

            advice.append(
                "The model indicates moderate approval likelihood. "
                "Review the suggested financial improvements."
            )

        else:

            advice.append(
                "The model indicates relatively high approval likelihood. "
                "Continue maintaining a healthy financial profile."
            )

    if not advice:

        advice.append(
            "No major improvement suggestions were identified."
        )

    return advice


# ============================================================
# 8. COUNTERFACTUAL LOAN PREDICTION
# ============================================================

def generate_counterfactuals(
    applicant,
    current_loan_amount,
    current_loan_term
):

    scenarios = []

    reductions = [
        1.00,
        0.90,
        0.80,
        0.70,
        0.60,
        0.50
    ]

    possible_terms = sorted(set([

        int(current_loan_term),

        min(int(current_loan_term) + 60, 480),

        min(int(current_loan_term) + 120, 480),

        480

    ]))

    for reduction in reductions:

        new_amount = max(
            1,
            int(round(
                current_loan_amount * reduction
            ))
        )

        for new_term in possible_terms:

            candidate = applicant.copy()

            candidate["LoanAmountlog"] = np.log1p(
                new_amount
            )

            candidate["Loan_Amount_Term_log"] = np.log1p(
                new_term
            )

            candidate_df = pd.DataFrame(
                [candidate]
            )[feature_columns]

            prediction = int(
                model.predict(candidate_df)[0]
            )

            approval_probability = None

            if hasattr(model, "predict_proba"):

                classes = list(model.classes_)

                if 1 in classes:

                    approval_probability = float(
                        model.predict_proba(candidate_df)[0][
                            classes.index(1)
                        ]
                    )

            scenarios.append({

                "Loan Amount (thousands)": new_amount,

                "Loan Term (months)": new_term,

                "Predicted Result": (
                    "Approved"
                    if prediction == 1
                    else "Rejected"
                ),

                "Approval Probability (%)": (
                    round(
                        approval_probability * 100,
                        2
                    )
                    if approval_probability is not None
                    else np.nan
                )

            })

    results = pd.DataFrame(scenarios)

    successful = results[
        results["Predicted Result"] == "Approved"
    ].copy()

    if not successful.empty:

        successful["Amount Change"] = (
            current_loan_amount
            - successful["Loan Amount (thousands)"]
        )

        successful["Term Change"] = (
            successful["Loan Term (months)"]
            - current_loan_term
        )

        successful["Change Score"] = (

            successful["Amount Change"].clip(lower=0)
            / max(current_loan_amount, 1)

            +

            successful["Term Change"].clip(lower=0)
            / 480

        )

        best = successful.sort_values(
            [
                "Change Score",
                "Approval Probability (%)"
            ],
            ascending=[True, False]
        ).iloc[0]

    else:

        best = None

    return results, best


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Quick Start")

    st.caption(
        "Load a sample profile to test the model."
    )

    for preset_name in PRESETS:

        st.button(
            preset_name,
            on_click=apply_preset,
            args=(preset_name,),
            use_container_width=True
        )

    st.button(
        "Reset form",
        on_click=reset_form,
        use_container_width=True
    )

    st.divider()

    st.header("About")

    st.write(
        "This application uses a trained machine-learning model "
        "to estimate loan approval."
    )

    st.caption(
        "Predictions are statistical estimates, not official bank decisions."
    )

    if st.session_state["history"]:

        st.divider()

        st.header("Recent Predictions")

        for item in reversed(
            st.session_state["history"][-5:]
        ):

            st.write(
                f"**{item['status']}** · "
                f"{item['prob']:.0f}% · {item['time']}"
            )

        if st.button(
            "Clear history",
            use_container_width=True
        ):

            st.session_state["history"] = []

            st.rerun()


# ============================================================
# 10. HERO BANNER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Loan Approval Predictor</h1>
        <p>
            Evaluate applicants, understand predictions,
            and explore alternative loan scenarios.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 11. APPLICANT INPUTS
# ============================================================

left, right = st.columns(
    [1.35, 1],
    gap="large"
)

with left:

    tab_personal, tab_financial = st.tabs(
        ["Personal", "Financial"]
    )

    with tab_personal:

        c1, c2 = st.columns(2)

        with c1:

            st.selectbox(
                "Gender",
                ["Male", "Female"],
                key="gender"
            )

            st.radio(
                "Married",
                ["Yes", "No"],
                key="married",
                horizontal=True
            )

            st.select_slider(
                "Number of Dependents",
                options=["0", "1", "2", "3+"],
                key="dependents"
            )

        with c2:

            st.selectbox(
                "Education",
                ["Graduate", "Not Graduate"],
                key="education"
            )

            st.radio(
                "Self Employed",
                ["No", "Yes"],
                key="self_employed",
                horizontal=True
            )

            st.selectbox(
                "Property Area",
                ["Urban", "Semiurban", "Rural"],
                key="property_area"
            )

    with tab_financial:

        c3, c4 = st.columns(2)

        with c3:

            st.number_input(
                "Applicant Income (per month)",
                min_value=0,
                step=500,
                key="applicant_income"
            )

            st.number_input(
                "Coapplicant Income (per month)",
                min_value=0,
                step=500,
                key="coapplicant_income"
            )

            st.radio(
                "Credit History",
                ["Good", "Bad"],
                key="credit_history",
                horizontal=True
            )

        with c4:

            st.slider(
                "Loan Amount (in thousands)",
                min_value=1,
                max_value=700,
                step=5,
                key="loan_amount"
            )

            st.slider(
                "Loan Term (months)",
                min_value=12,
                max_value=480,
                step=12,
                key="loan_term"
            )

            st.slider(
                "Assumed interest rate (% p.a.)",
                min_value=1.0,
                max_value=20.0,
                step=0.25,
                key="interest_rate"
            )


# ============================================================
# 12. LIVE EMI AND FINANCIAL SUMMARY
# ============================================================

ss = st.session_state

total_income = (
    ss["applicant_income"] +
    ss["coapplicant_income"]
)

loan_value = ss["loan_amount"] * 1000

emi = calculate_emi(
    loan_value,
    ss["loan_term"],
    ss["interest_rate"]
)

emi_ratio = (
    emi / total_income * 100
    if total_income > 0
    else None
)

loan_to_income = (
    loan_value / (total_income * 12)
    if total_income > 0
    else None
)

with right:

    st.markdown("#### Live Summary")

    m1, m2 = st.columns(2)

    m1.metric(
        "Total monthly income",
        f"{total_income:,.0f}"
    )

    m2.metric(
        "Loan amount",
        f"{loan_value:,.0f}"
    )

    m3, m4 = st.columns(2)

    m3.metric(
        "Estimated EMI",
        f"{emi:,.0f}"
    )

    m4.metric(
        "EMI / income",
        f"{emi_ratio:.0f}%"
        if emi_ratio is not None
        else "—"
    )

    if emi_ratio is not None:

        st.progress(
            min(emi_ratio / 100, 1.0),
            text="Monthly burden on income"
        )

        if emi_ratio > 50:

            st.warning(
                "EMI is over 50% of income."
            )

        elif emi_ratio > 35:

            st.info(
                "EMI is moderately high relative to income."
            )

        else:

            st.success(
                "EMI looks comfortable relative to income."
            )

    else:

        st.warning(
            "Enter an income to see affordability."
        )

    st.caption(
        "EMI is an estimate. The assumed interest rate is not "
        "used by the prediction model."
    )


# ============================================================
# 13. RUN PREDICTION
# ============================================================

st.divider()

if st.button(
    "Predict Loan Status",
    type="primary",
    use_container_width=True
):

    if total_income <= 0:

        st.error(
            "Enter a positive total income."
        )

    elif ss["loan_amount"] <= 0:

        st.error(
            "Loan amount must be greater than zero."
        )

    else:

        applicant = build_applicant(ss)

        try:

            input_df, prediction, probability = (
                predict_applicant(applicant)
            )

            result = {

                "applicant": applicant,

                "input_df": input_df,

                "prediction": prediction,

                "probability": probability,

                "approved": prediction == 1,

                "status": (
                    "APPROVED"
                    if prediction == 1
                    else "REJECTED"
                ),

                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),

                "inputs": {
                    key: ss[key]
                    for key in DEFAULTS
                }

            }

            st.session_state["last_result"] = result

            st.session_state["counterfactual_results"] = None

            st.session_state["history"].append({

                "status": result["status"],

                "prob": (
                    probability * 100
                    if probability is not None
                    else (
                        100.0
                        if result["approved"]
                        else 0.0
                    )
                ),

                "time": datetime.now().strftime("%H:%M")

            })

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# 14. DISPLAY PREDICTION RESULT
# ============================================================

result = st.session_state.get("last_result")

if result is not None:

    applicant = result["applicant"]

    input_df = result["input_df"]

    prediction = result["prediction"]

    prob = result["probability"]

    approved = result["approved"]

    status = result["status"]

    # Calculate applicant category
    risk_category, risk_description = (
        categorize_applicant(prob)
    )

    st.divider()

    st.subheader("Prediction Result")

    res_col, detail_col = st.columns(
        [1, 1.2],
        gap="large"
    )

    with res_col:

        if approved:

            st.markdown(
                """
                <div class="result-approved">
                    <h2>APPROVED</h2>
                    <p>
                        The model predicts this applicant
                        is likely to be approved.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-rejected">
                    <h2>REJECTED</h2>
                    <p>
                        The model predicts this applicant
                        is likely to be rejected.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        if prob is not None:

            meter_color = (
                "#38ef7d"
                if prob >= 0.60
                else "#f5b041"
                if prob >= 0.40
                else "#ef473a"
            )

            st.markdown(
                f"**Approval probability: {prob * 100:.1f}%**"
            )

            st.markdown(
                f"""
                <div class="meter-wrap">
                    <div class="meter-fill"
                         style="width:{prob * 100:.1f}%;
                                background:{meter_color};">
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with detail_col:

        st.markdown("#### What stands out")

        notes = []

        if ss["credit_history"] == "Bad":

            notes.append(
                "Credit-history value is marked as 0."
            )

        else:

            notes.append(
                "Credit-history value is marked as 1."
            )

        if loan_to_income is not None:

            if loan_to_income > 4:

                notes.append(
                    f"Loan amount is {loan_to_income:.1f} "
                    "times annual income."
                )

            else:

                notes.append(
                    f"Loan amount is {loan_to_income:.1f} "
                    "times annual income."
                )

        if emi_ratio is not None and emi_ratio > 40:

            notes.append(
                f"Estimated EMI takes {emi_ratio:.0f}% "
                "of monthly income."
            )

        if ss["coapplicant_income"] > 0:

            notes.append(
                "Co-applicant income contributes to household income."
            )

        for note in notes:
            st.write(note)

        st.caption(
            "These are descriptive observations, not explanations "
            "of the model's exact reasoning."
        )


    # ========================================================
    # 15. APPLICANT RISK CATEGORIZATION
    # ========================================================

    st.divider()

    st.subheader(
        "Applicant Risk Categorization"
    )

    st.write(
        "This feature categorizes applicants according to "
        "their predicted loan approval probability."
    )

    st.caption(
        "The thresholds are defined for this academic project. "
        "They are not official banking standards and do not "
        "measure actual loan default risk."
    )

    if risk_category == "Low":

        st.success(
            f"LOW INDICATOR — {risk_description}"
        )

    elif risk_category == "Medium":

        st.warning(
            f"MEDIUM INDICATOR — {risk_description}"
        )

    elif risk_category == "High":

        st.error(
            f"HIGH INDICATOR — {risk_description}"
        )

    else:

        st.info(risk_description)

    r1, r2 = st.columns(2)

    r1.metric(
        "Applicant Category",
        risk_category
    )

    r2.metric(
        "Approval Probability",
        f"{prob * 100:.1f}%"
        if prob is not None
        else "Unavailable"
    )


    # ========================================================
    # 16. MODEL FEATURE IMPORTANCE
    # ========================================================

    if hasattr(model, "feature_importances_"):

        with st.expander(
            "Model Feature Importance"
        ):

            importance = pd.Series(
                model.feature_importances_,
                index=feature_columns
            ).sort_values()

            st.bar_chart(
                importance,
                horizontal=True
            )


    # ========================================================
    # 17. SHAP EXPLAINABLE AI
    # ========================================================

    st.divider()

    st.subheader(
        "Explainable AI - SHAP Analysis"
    )

    st.write(
        "SHAP explains how individual features contribute "
        "to the model's predictions."
    )

    try:

        with st.spinner(
            "Generating SHAP explanations..."
        ):

            explainer, background_data = load_shap_tools()

            individual_raw = explainer.shap_values(
                input_df
            )

            individual_values = shap_class_one_values(
                individual_raw
            )[0]

            base_value = shap_class_one_base(
                explainer
            )

            waterfall = shap.Explanation(

                values=individual_values,

                base_values=base_value,

                data=input_df.iloc[0].values,

                feature_names=feature_columns

            )

            st.markdown(
                "#### Individual Applicant Explanation"
            )

            st.caption(
                "The waterfall plot shows how features contribute "
                "to the model output for this applicant."
            )

            shap.plots.waterfall(
                waterfall,
                max_display=len(feature_columns),
                show=False
            )

            st.pyplot(
                plt.gcf(),
                clear_figure=True
            )

            plt.close("all")

            # Global SHAP values
            global_raw = explainer.shap_values(
                background_data
            )

            global_values = shap_class_one_values(
                global_raw
            )

            # Global feature importance
            st.markdown(
                "#### Global SHAP Feature Importance"
            )

            st.caption(
                "Features are ranked by average absolute SHAP value."
            )

            shap.summary_plot(
                global_values,
                background_data,
                plot_type="bar",
                show=False
            )

            st.pyplot(
                plt.gcf(),
                clear_figure=True
            )

            plt.close("all")

            # SHAP summary plot
            st.markdown(
                "#### SHAP Summary Plot"
            )

            st.caption(
                "This plot displays the distribution and direction "
                "of feature contributions."
            )

            shap.summary_plot(
                global_values,
                background_data,
                show=False
            )

            st.pyplot(
                plt.gcf(),
                clear_figure=True
            )

            plt.close("all")

    except Exception as error:

        st.warning(
            "SHAP could not generate its explanation. "
            "The prediction is still available. Check Dataset.csv "
            "and the installed SHAP/scikit-learn versions."
        )

        st.exception(error)


    # ========================================================
    # 18. LOAN ELIGIBILITY IMPROVEMENT ADVISOR
    # ========================================================

    st.divider()

    st.subheader(
        "Loan Eligibility Improvement Advisor"
    )

    st.write(
        "This advisor provides general suggestions based on "
        "the applicant's financial details."
    )

    credit_encoded = (
        1
        if ss["credit_history"] == "Good"
        else 0
    )

    advice_list = generate_eligibility_advice(

        applicant_income=ss["applicant_income"],

        coapplicant_income=ss["coapplicant_income"],

        loan_amount_thousands=ss["loan_amount"],

        loan_term=ss["loan_term"],

        credit_history=credit_encoded,

        approval_probability=prob,

        emi_ratio=emi_ratio

    )

    for advice in advice_list:

        st.info(
            advice
        )


    # ========================================================
    # 19. COUNTERFACTUAL LOAN PREDICTION
    # ========================================================

    st.divider()

    st.subheader(
        "Counterfactual Loan Prediction"
    )

    st.write(
        "Explore alternative loan amounts and repayment periods "
        "while keeping the other applicant details unchanged."
    )

    st.caption(
        "These are hypothetical model scenarios, not guarantees "
        "of loan approval."
    )

    if st.button(
        "Find Alternative Loan Options",
        use_container_width=True
    ):

        with st.spinner(
            "Testing alternative loan scenarios..."
        ):

            scenarios, best_option = generate_counterfactuals(

                applicant=applicant,

                current_loan_amount=ss["loan_amount"],

                current_loan_term=ss["loan_term"]

            )

            st.session_state["counterfactual_results"] = {

                "scenarios": scenarios,

                "best": best_option

            }

    counterfactual_data = st.session_state.get(
        "counterfactual_results"
    )

    if counterfactual_data is not None:

        best = counterfactual_data["best"]

        scenarios = counterfactual_data["scenarios"]

        if best is not None:

            st.success(
                "The model found an alternative scenario "
                "predicted as approved."
            )

            cf1, cf2, cf3 = st.columns(3)

            cf1.metric(
                "Suggested Loan Amount",
                f"{best['Loan Amount (thousands)'] * 1000:,.0f}"
            )

            cf2.metric(
                "Suggested Loan Term",
                f"{int(best['Loan Term (months)'])} months"
            )

            if pd.notna(
                best["Approval Probability (%)"]
            ):

                cf3.metric(
                    "Approval Probability",
                    f"{best['Approval Probability (%)']:.1f}%"
                )

            st.info(
                "This is a hypothetical model scenario, not "
                "a financial recommendation or approval guarantee."
            )

        else:

            st.warning(
                "None of the tested alternative scenarios "
                "produced an approval prediction."
            )

        st.markdown(
            "#### Tested Alternative Scenarios"
        )

        st.dataframe(
            scenarios,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # 20. DOWNLOADABLE PREDICTION REPORT
    # ========================================================

    st.divider()

    st.subheader(
        "Download Prediction Report"
    )

    report_lines = [

        f"Loan Prediction Report - {result['timestamp']}",

        "-" * 48,

        f"Prediction: {status}",

        (
            f"Approval Probability: {prob * 100:.1f}%"
            if prob is not None
            else "Approval Probability: Unavailable"
        ),

        f"Applicant Category: {risk_category}",

        f"Category Explanation: {risk_description}",

        "",

        "Applicant Details",

        f"Gender: {ss['gender']}",

        f"Married: {ss['married']}",

        f"Dependents: {ss['dependents']}",

        f"Education: {ss['education']}",

        f"Self Employed: {ss['self_employed']}",

        f"Property Area: {ss['property_area']}",

        f"Applicant Income: {ss['applicant_income']}",

        f"Coapplicant Income: {ss['coapplicant_income']}",

        f"Loan Amount (thousands): {ss['loan_amount']}",

        f"Loan Term (months): {ss['loan_term']}",

        f"Credit History: {ss['credit_history']}",

        "",

        "Eligibility Improvement Suggestions"

    ]

    report_lines.extend([
        f"- {item}"
        for item in advice_list
    ])

    if counterfactual_data is not None:

        report_lines.extend([

            "",

            "Counterfactual Scenario Results",

            counterfactual_data["scenarios"].to_string(
                index=False
            )

        ])

    report_lines.extend([

        "",

        "Disclaimer:",

        "This is a machine-learning prediction for an academic project. "
        "It is not an official bank decision or a guarantee of approval.",

        "The applicant category represents predicted approval likelihood, "
        "not actual loan default risk."

    ])

    report_text = "\n".join(
        report_lines
    )

    st.download_button(

        "Download Prediction Report",

        data=report_text,

        file_name="loan_prediction_report.txt",

        mime="text/plain",

        use_container_width=True

    )

    st.caption(
        "The model estimates loan approval based on its training data. "
        "It does not predict actual loan default."
    )