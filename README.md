# Online Shoppers Purchasing Intention — Machine Learning Case Study

This repository presents a machine learning case study focused on predicting purchase intent using detailed session-level browsing behavior.  
The work is based on the **Online Shoppers Purchasing Intention** dataset from the UCI Machine Learning Repository.

### Notebook  
The full analysis, including exploration, feature preparation, modeling, and interpretability, is documented here:  
**https://github.com/movet306/Online-Shoppers-Purchasing-Intention-Machine-Learning-Case-Study/blob/main/Mert_Ovet%20Online%20Shoppers%20Purchasing%20Intention.ipynb**

---

## Project Overview

The goal of this study is to examine how user actions—page interactions, engagement depth, bounce/exit behavior, technical metadata, and seasonality—relate to the likelihood of completing a purchase.

The project follows a practical machine-learning structure: understand the data, engineer stable features, compare model behavior, and interpret how the model reaches its decisions.

---

## 1. Data Understanding

The dataset contains information on:

- Counts and durations of administrative, informational, and product-related pages  
- Bounce and exit rates  
- Page value indicators  
- Browser, operating system, region, traffic source  
- Month and weekend flags  
- **Revenue** label (purchase vs. non-purchase)

EDA shows that behavioral metrics (engagement depth, bounce/exit behavior, product­-related activity) carry the strongest signal, while most categorical variables have limited separation power.

---

## 2. Feature Preparation

The workflow includes:

- Winsorization of heavy right-tail outliers  
- `log1p` transformation for skew reduction  
- One-hot encoding of categorical variables  
- Scaling of transformed numeric features  
- Stratified train–test split to preserve the 85/15 class ratio  

All preprocessing steps were checked for alignment, leakage, duplicates, or structural errors.

---

## 3. Modeling

Two model families were assessed:

### Logistic Regression
- High recall, structurally low precision  
- Hyperparameter tuning produced no meaningful gain  
- Limited by linear boundaries and weak categorical structure

### Random Forest
- Stronger handling of nonlinear patterns  
- Better F1 performance  
- Improved recall compared with baseline  
- Threshold exploration conducted to understand trade-offs  

The tuned Random Forest achieved the most balanced metrics and became the primary candidate for interpretation.

---

## 4. Interpretability

Multiple interpretability methods were applied:

- Feature importance (Gini)  
- Permutation importance  
- Partial dependence plots  
- ROC curve  
- Precision–Recall curve  
- Confusion matrix analysis  

Findings:

- Behavioral metrics dominate the model’s decisions.  
- Seasonal patterns—especially November and late-Q4 activity—meaningfully influence predictions.  
- Browser, OS, Region, and similar technical attributes contribute only marginally.

---

## 5. Final Model Selection

The **F1-tuned Random Forest** with a threshold near **0.50** is the recommended configuration.

Reasons:

- More reliable separation of purchasers vs. non-purchasers  
- Strong alignment with behavioral patterns  
- Stable performance across metrics  
- Minimal sensitivity to threshold shifts near the optimum  

The model is well-suited for broad-audience lead identification and session-level purchase-intent scoring, especially during seasonal peaks.

---

## 6. Limitations and Future Work

The dataset has several constraints:

- No multi-session user history  
- No fine-grained click-level sequencing  
- Possible target leakage through PageValues  
- Persistent class imbalance and noisy negatives  
- Limited temporal granularity (no hour-of-day, weekday patterns)

---
