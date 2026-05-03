# ML/AI Internship Learning Journey
### Divya Vemana | Started: April 6, 2025 | Zenitude Internship

---

## Overview

This repository documents my complete learning journey through the 8-week ML/AI internship program. Each folder corresponds to a week in the plan and contains code, notebooks, and detailed documentation of everything I built and learned.

> I focused on deeply understanding and implementing each concept . Every project has a thorough README explaining my approach, the reasoning behind each decision

---

## Progress Tracker

| Week | Theme | Status | Key Deliverables |
|------|-------|--------|-----------------|
| Week 1 | Python & Math Foundations | ✅ Complete | Kaggle Pandas course, LeetCode 50+, NumPy/Pandas/Stats |
| Week 2 | Data Analysis & Visualization | ✅ Complete | Titanic EDA, HackerRank SQL Intermediate Certificate |
| Week 3 | Classical Machine Learning | ✅ Complete | Titanic Kaggle 0.782, House Prices R²=0.923, Mall Segmentation |
| Week 4 | Deep Learning Fundamentals | 🔄 In Progress | CIFAR-10 classifier, Andrew Ng Course 1 |
| Week 5 | NLP & Advanced DL | ⏳ Upcoming | |
| Week 6 | Full-Stack for ML | ⏳ Upcoming | |
| Week 7 | MLOps & Cloud | ⏳ Upcoming | |
| Week 8 | Portfolio Polish | ⏳ Upcoming | |

---

## Repository Structure

```
ML-Internship-Journey/
│
├── Week1-Python-Math-Foundations/
│   └── README.md
│
├── Week2-EDA-DataAnalysis/
│   └── README.md
│
├── Week3-Classical-ML/
│   ├── Titanic-Survival-Prediction/
│   │   ├── titanic_notebook.ipynb
│   │   └── README.md
│   ├── House-Prices-Regression/
│   │   ├── house_prices_notebook.ipynb
│   │   └── README.md
│   └── Mall-Customer-Segmentation/
│       ├── mall_customers_notebook.ipynb
│       └── README.md
│
└── Week4-Deep-Learning/ (in progress)
```

---

## Projects

### Titanic Survival Prediction
**Week 2 + 3 | Classification | Kaggle Public Score: 0.78229**

Complete end-to-end ML pipeline from raw data to Kaggle leaderboard.

- Full EDA with 17+ visualizations covering missing values, distributions, survival patterns
- Feature Engineering — Title extracted from Name, FamilySize, IsAlone, AgeBand, FareBand
- Models: Logistic Regression, Random Forest, XGBoost, Voting Ensemble
- Hyperparameter tuning with GridSearchCV
- Evaluation: Confusion Matrix, F1 (0.795), ROC-AUC (0.903)
- Kaggle Public Score: **0.78229**



---

### House Prices — Advanced Regression
**Week 3 | Regression | Validation R²: 0.923**

End-to-end regression on the Ames Housing dataset (79 features).

- Log transformed skewed target (skewness 1.88 → 0.12)
- Domain-informed missing value handling across 19 columns
- Feature Engineering — TotalSF, HouseAge, QualLivArea interaction feature
- Fixed 236 skewed numerical features
- Models: Linear Regression, Ridge, Lasso, Random Forest, XGBoost
- Final blend: Ridge(25%) + Lasso(35%) + XGBoost(40%)
- Validation R²: **0.923**


---

### Mall Customer Segmentation
**Week 3 | Unsupervised Learning | Silhouette Score: 0.5547**

Complete unsupervised learning project — K-Means, PCA, DBSCAN, SVM, RandomizedSearchCV.

- K-Means with Elbow Method + Silhouette Score → optimal K=5
- 5 business-meaningful segments: Target Customers, Impulsive Buyers, Careful Spenders, Budget Customers, Regular Customers
- PCA — 77.6% variance explained in 2 components, analysed feature loadings
- DBSCAN — identified 35 outlier customers (17.5%) who don't fit any segment
- SVM — 97.5% accuracy classifying customers into segments
- RandomizedSearchCV — honest finding that default parameters were already optimal


---

## Skills Covered

**Supervised Learning**
Linear Regression, Ridge, Lasso, Logistic Regression, Random Forest, XGBoost, SVM, Decision Trees

**Unsupervised Learning**
K-Means Clustering, PCA, DBSCAN

**Model Evaluation**
Cross-validation (5-fold), GridSearchCV, RandomizedSearchCV, Confusion Matrix, F1 Score, Precision, Recall, ROC-AUC, RMSE, R²

**Data & Tools**
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Plotly, Jupyter

**Other**
SQL (HackerRank Intermediate Certificate), EDA, Feature Engineering, Kaggle Competitions

---

## Kaggle

- Titanic — Machine Learning from Disaster: **0.78229** public score
- House Prices — Advanced Regression Techniques: Submitted

---

## Week Summaries

### Week 1 — Python & Math Foundations
Completed Kaggle Pandas micro-course, revised Python OOP, NumPy, linear algebra and statistics fundamentals, solved 50+ LeetCode Easy problems. The statistics work (distributions, correlation, hypothesis testing) directly applied in Weeks 2 and 3.

[View Week 1 →](./Week1-Python-Math-Foundations/)

### Week 2 — Data Analysis & Visualization
Completed full EDA on Titanic dataset using Matplotlib, Seaborn and Plotly. Earned HackerRank SQL Intermediate Certificate covering window functions, CTEs, subqueries and complex joins. The EDA findings directly shaped the feature engineering approach in Week 3.

[View Week 2 →](./Week2-EDA-DataAnalysis/)

### Week 3 — Classical Machine Learning
Built 3 end-to-end ML projects covering the full syllabus — supervised classification, regression, and unsupervised learning. Submitted to 2 Kaggle competitions. Key learning: feature engineering consistently had more impact on model performance than algorithm choice.

[View Week 3 →](./Week3-Classical-ML/)