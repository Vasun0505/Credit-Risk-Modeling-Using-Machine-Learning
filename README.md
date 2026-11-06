# Credit Risk Modelling Using Machine Learning

### An Explainable Machine Learning Approach to Loan Approval Prediction

---

## Overview

Credit Risk Modelling is a machine learning-based project focused on analysing loan applicant information and predicting loan approval outcomes. The system uses applicant financial details, credit history and demographic attributes to identify patterns associated with loan approval decisions.

The project implements multiple classification algorithms and evaluates their predictive performance to identify a suitable model. It further integrates Explainable AI techniques, including SHAP and Counterfactual Analysis, to improve the interpretability of model predictions.

An interactive Streamlit dashboard provides a unified interface for prediction, risk categorisation and model explanations.

## Problem Definition

Traditional loan evaluation involves examining multiple applicant attributes before reaching a lending decision. Identifying relevant patterns across these attributes can be challenging when processing large volumes of applications.

This project investigates the application of machine learning techniques to automate loan approval prediction while providing interpretable insights into the factors influencing the model's decisions.

## Objectives

- Perform statistical analysis and exploratory data analysis on loan applicant data.
- Apply data preprocessing techniques to improve data quality.
- Engineer relevant features to enhance model representation.
- Develop and evaluate multiple classification algorithms.
- Identify the best-performing model using appropriate evaluation metrics.
- Integrate explainability techniques to interpret individual predictions.
- Develop an interactive dashboard for real-time applicant assessment.

## System Workflow

```text
          Loan Dataset
                |
                v
       Data Preprocessing
                |
                v
       Exploratory Analysis
                |
                v
       Feature Engineering
                |
                v
       Data Splitting & Balancing
                |
                v
        Model Development
                |
                v
        Model Evaluation
                |
                v
        Random Forest Model
                |
         +------+------+
         |             |
         v             v
    SHAP Analysis   Counterfactual
         |             |
         +------+------+
                |
                v
       Streamlit Dashboard
                |
                v
       Prediction & Insights
```

## Dataset Description

The project uses a loan approval dataset containing **614 records and 13 attributes**.

The dataset includes applicant information such as:

- Demographic details: Gender, Married, Dependents and Education.
- Financial information: Applicant Income, Co-applicant Income and Loan Amount.
- Loan details: Loan Amount Term and Credit History.
- Property information: Property Area.
- Target variable: Loan Status.

The target variable contains two classes:

- `Y` – Loan Approved
- `N` – Loan Rejected

## Methodology

### 1. Data Preprocessing
Missing values are handled using median imputation for numerical attributes and mode imputation for categorical attributes. Duplicate records and inconsistent values are also examined.

### 2. Exploratory Data Analysis
Statistical summaries, distribution plots, boxplots, categorical comparisons and correlation heatmaps are used to identify patterns and relationships within the dataset.

### 3. Feature Engineering
Additional features are generated, including `Total_Income`. Logarithmic transformations are applied to selected financial attributes to reduce the influence of skewed distributions.

### 4. Data Balancing
The dataset is divided into training and testing subsets. Random oversampling is applied to the training data to address class imbalance.

### 5. Model Development
Four classification algorithms are implemented and compared:

| Algorithm | Approach |
|---|---|
| Logistic Regression | Probabilistic classification |
| Decision Tree | Rule-based classification |
| K-Nearest Neighbours | Distance-based classification |
| Random Forest | Ensemble-based classification |

### 6. Model Evaluation
Model performance is assessed using Accuracy, Precision, Recall, F1-Score, AUC and cross-validation.

## Model Performance

The following results were obtained from the held-out test dataset:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 75.32% | 75.41% | 92.00% | 82.88% | 0.70 |
| Decision Tree | 65.58% | 73.74% | 73.00% | 73.37% | 0.62 |
| KNN | 67.53% | 75.51% | 74.00% | 74.75% | 0.66 |
| **Random Forest** | **76.62%** | **77.12%** | **91.00%** | **83.49%** | **0.78** |

Random Forest demonstrated the strongest overall performance, achieving the highest Accuracy, F1-Score and AUC among the evaluated models. It was therefore selected as the preferred prediction model.

## Explainable AI Integration

A key component of this project is the integration of explainability techniques to provide insights into model behaviour.

**SHAP (SHapley Additive exPlanations)**

SHAP is used to quantify the contribution of individual input features towards model predictions. It supports both global feature importance analysis and local explanations for individual applicants.

**Counterfactual Analysis**

Counterfactual analysis explores alternative applicant input values that may change the model's predicted outcome. It provides insight into possible changes in loan-related attributes, such as the requested loan amount or repayment term.

## Application Features

The Streamlit dashboard integrates the trained prediction model with the explainability components.

| Component | Functionality |
|---|---|
| Loan Prediction | Generates approval or rejection predictions |
| Probability Analysis | Displays predicted approval probability |
| Risk Categorisation | Groups predictions into Low, Medium and High categories |
| SHAP Explanation | Identifies influential applicant attributes |
| Counterfactual Analysis | Explores alternative inputs and prediction outcomes |
| Eligibility Insights | Presents prediction-related guidance |

## Technology Stack

| Category | Tools and Technologies |
|---|---|
| Programming Languages | R, Python |
| Data Manipulation | dplyr, tidyr |
| Data Visualisation | ggplot2 |
| Machine Learning | caret, randomForest, rpart, class |
| Explainable AI | SHAP |
| Application Framework | Streamlit |
| Development Tools | RStudio, VS Code |
| Version Control | Git, GitHub |

## Repository Structure

The repository will contain the following project resources as development and documentation files are uploaded:

```text
Credit-Risk-Modeling-Using-Machine-Learning/
│
├── README.md
├── .gitignore
├── dataset/
├── R/
├── app/
├── models/
└── documentation/
```

## Future Scope

- Training and validation using larger and more diverse datasets.
- Further optimisation of model performance and classification thresholds.
- Additional explainability and fairness evaluation.
- Integration of more advanced machine learning algorithms.
- Enhancement of dashboard functionality and visual analytics.

## Limitations and Disclaimer

The model is developed using a historical loan approval dataset. Its predictions reflect patterns learned from the available data and may not represent actual lending decisions.

The application is intended for academic and educational purposes and should not be considered a substitute for professional financial assessment.

## Project Repository

**GitHub:** [Credit Risk Modelling Using Machine Learning](https://github.com/Vasun0505/Credit-Risk-Modeling-Using-Machine-Learning)

---

*Academic Project | Data Science | Machine Learning | Explainable AI*
