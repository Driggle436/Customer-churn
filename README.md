# Telco Customer Churn Prediction

## Project Overview

Customer churn is a major business problem for telecommunications companies. Identifying customers who are likely to leave allows companies to take proactive retention actions.

This project develops a machine learning pipeline to predict customer churn and segment customers according to their estimated churn risk.

The project covers exploratory data analysis, data preprocessing, machine learning model comparison, probability threshold optimization, and customer risk segmentation.

---

##  Problem Statement

The objective is to predict whether a telecom customer is likely to churn based on demographic information, services, contract details, payment method, tenure, and billing information.

The project also aims to identify the characteristics associated with higher churn risk and translate the model's predictions into actionable business recommendations.

---

##  Dataset

The project uses the **Telco Customer Churn** dataset from Kaggle.

The dataset contains:

* **7,043 customers**
* **20 features**
* Customer demographics
* Account information
* Services subscribed to
* Contract information
* Payment information
* Monthly and total charges
* Churn status

### Target Variable

`Churn`

* `No` → Customer retained
* `Yes` → Customer churned

---

##  Exploratory Data Analysis

Several important churn patterns were identified.

### Contract Type

| Contract       | Churn Rate |
| -------------- | ---------: |
| Month-to-month |     ~42.7% |
| One year       |     ~11.3% |
| Two year       |      ~2.8% |

Month-to-month customers showed substantially higher churn than customers on longer-term contracts.

### Tenure

| Tenure       | Churn Rate |
| ------------ | ---------: |
| 0–12 months  |     ~47.5% |
| 13–24 months |     ~28.6% |
| 25–48 months |     ~20.4% |
| 49–72 months |      ~9.6% |

Newer customers were considerably more likely to churn.

### Other Important Patterns

Higher observed churn was also associated with:

* Fiber-optic internet service
* Electronic check payment
* Paperless billing
* Lack of online security
* Lack of technical support
* Higher monthly charges
* Senior citizen status

These relationships represent associations in the dataset and should not be interpreted as proof of causation.

---

##  Data Preprocessing

The following preprocessing steps were performed:

* Converted `TotalCharges` to numerical format
* Replaced missing `TotalCharges` values with `0` for customers with zero tenure
* Converted the target variable into binary format
* Separated features and target
* Used an 80/20 stratified train-test split
* Standardized numerical features using `StandardScaler`
* One-hot encoded categorical variables
* Used `drop="first"` for categorical encoding
* Used a Scikit-learn `Pipeline` and `ColumnTransformer` to prevent preprocessing leakage

---

##  Machine Learning Models

Three classification models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Model Comparison

| Model                   |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression** | **80.70%** | **66.04%** | **56.15%** | **60.69%** | **84.22%** |
| Random Forest           |     78.35% |     62.11% |     47.33% |     53.72% |     82.27% |
| XGBoost                 |     80.34% |     66.55% |     52.14% |     58.47% | **84.23%** |

### Final Model

Logistic Regression was selected as the final model because it achieved the best overall balance of accuracy, recall, F1-score, and ROC-AUC while also providing strong interpretability.

Although XGBoost achieved a marginally higher ROC-AUC, the difference was negligible and Logistic Regression provided better recall and F1-score.

---

##  Threshold Optimization

The default classification threshold of 0.50 was compared with alternative thresholds.

| Threshold | Precision |     Recall |   F1 Score |
| --------: | --------: | ---------: | ---------: |
|      0.30 |    51.93% | **75.40%** | **61.50%** |
|      0.40 |    56.80% |     66.80% |     61.40% |
|      0.50 |    66.04% |     56.15% |     60.69% |

For a retention-focused use case, a threshold of **0.30** was selected as an operating threshold because it substantially improves recall.

This allows the business to identify a larger proportion of customers who are likely to churn, at the cost of generating more false positives.

---

##  Customer Risk Segmentation

Customers were segmented using predicted churn probabilities:

* **Low Risk:** probability < 0.30
* **Medium Risk:** 0.30–0.70
* **High Risk:** probability ≥ 0.70

### Risk Distribution

| Risk Level | Customers | Observed Churn Rate |
| ---------- | --------: | ------------------: |
| Low        |       866 |          **10.62%** |
| Medium     |       450 |          **47.33%** |
| High       |        93 |          **74.19%** |

The observed churn rate increased substantially across the risk groups, indicating that the model provides useful customer prioritization.

---

##  Business Recommendations

Based on the analysis, the telecom company could consider:

### 1. Focus on new customers

Customers in their first year showed substantially higher churn. A dedicated onboarding and early-retention program could help reduce early customer loss.

### 2. Encourage longer-term contracts

Month-to-month customers showed significantly higher churn than one- and two-year customers. Targeted incentives could encourage customers to move to longer-term contracts.

### 3. Prioritize high-risk customers

The high-risk segment had an observed churn rate of approximately 74%, making this group suitable for prioritized retention campaigns.

### 4. Investigate electronic-check customers

Electronic-check users showed substantially higher churn. The company could investigate whether payment experience, billing friction, or customer characteristics contribute to this pattern.

### 5. Improve support and security adoption

Customers without online security and technical support showed higher churn rates. Bundling or promoting these services could be explored as part of retention strategies.

---

##  Project Structure

```text
Customer-churn/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── Customer_churn.ipynb
│
├── README.md
│
└── .gitignore
```

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook
* Git & GitHub

---

##  Key Takeaway

The project demonstrates an end-to-end customer churn prediction workflow, from data cleaning and exploratory analysis to machine learning, threshold optimization, and business-oriented customer risk segmentation.

The final Logistic Regression model achieved an ROC-AUC of approximately **0.842** and provided an interpretable approach for prioritizing customers who may require retention intervention.

---

##  Dataset Source

Telco Customer Churn dataset available through Kaggle.
