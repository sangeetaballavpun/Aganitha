# Drug Prediction Using Machine Learning

## Problem Statement
A pharmaceutical company has collected clinical data from patients suffering from the same illness.
Each patient responded to one of five drugs: **Drug A, Drug B, Drug C, Drug X, and Drug Y**.
The objective of this project is to build a machine learning model that can predict the most suitable
drug for a future patient based on demographic and clinical features.

This project demonstrates a complete machine learning workflow including exploratory data analysis,
model development, evaluation, and interpretation.

---

## Dataset Description
The dataset contains **200 patient records** with the following features:

- **Age** – Numerical  
- **Sex** – Categorical (Male / Female)  
- **Blood Pressure (BP)** – Categorical (Low / Normal / High)  
- **Cholesterol** – Categorical (Normal / High)  
- **Target Variable** – Drug (A, B, C, X, Y)

The dataset is clean and contains **no missing values**.

---

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis was performed to understand feature distributions and their relationship
with the target variable. Key observations include:

- Blood Pressure is the strongest predictor of drug selection
- Cholesterol is the second most influential feature
- Age acts as a secondary modifier
- Sex has minimal independent impact

Clear rule-based and non-linear patterns were observed, indicating that tree-based models are well
suited for this problem.

---

## Data Preprocessing & Feature Engineering
- Categorical variables (Sex, BP, Cholesterol, Drug) were encoded using **Label Encoding**
- Data was split into training (80%) and testing (20%) sets
- **Stratified sampling** was used to preserve class distribution
- No missing value imputation was required

---

## Model Development
Three classification models were trained and evaluated:

### 1. Logistic Regression (Baseline)
Logistic Regression was used as a baseline model due to its simplicity and interpretability.
However, it assumes linear decision boundaries and struggles to capture complex,
rule-based relationships present in the data.

A higher number of iterations was used to ensure convergence.

### 2. Decision Tree
Decision Tree was used to capture hierarchical and non-linear decision rules.
It offers strong interpretability, making it suitable for clinical reasoning.

### 3. Random Forest
Random Forest, an ensemble of multiple decision trees, was implemented to improve robustness
and reduce overfitting. It consistently achieved the best performance across evaluation metrics.

---

## Hyperparameter Tuning
Hyperparameter tuning was performed on the best-performing model (**Random Forest**) using
**GridSearchCV**. Parameters such as the number of estimators, tree depth, and minimum samples
for splitting were optimized using 5-fold cross-validation.

---

## Model Evaluation
Given the multi-class nature of the problem, evaluation was not limited to accuracy.

The following metrics were used:
- Accuracy
- Precision, Recall, and F1-score (Classification Report)
- Confusion Matrix
- **Macro-averaged ROC-AUC** (One-vs-Rest strategy)
- 5-fold Cross-Validation Accuracy

Macro-averaged metrics were used to ensure fair evaluation across all drug classes,
accounting for mild class imbalance.

---

## Why Logistic Regression Underperforms
Logistic Regression underperforms compared to tree-based models because:
- It assumes linear separability between classes
- The dataset exhibits non-linear, rule-based patterns
- Interactions between Blood Pressure and Cholesterol are better captured by trees

This makes tree-based models more suitable for this problem.

---

## Feature Importance
Feature importance analysis from the tuned Random Forest model revealed:
1. Blood Pressure – most critical feature
2. Cholesterol – second most important
3. Age – moderate influence
4. Sex – least influence

These results align well with clinical intuition.

---

## Final Recommendation
**Random Forest** is recommended for deployment in a clinical decision-support setting because:
- It achieved the highest cross-validated accuracy
- It showed strong macro-averaged ROC-AUC
- It is robust to class imbalance
- It captures non-linear clinical decision rules
- Feature importance provides interpretability

---

## How to Run the Notebook

### Option 1: Local Jupyter (Recommended)
```bash
pip install -r requirements.txt
jupyter notebook
