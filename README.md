# ML/AI Internship Learning Journey
### Divya Vemana | Started: April 6, 2025 | Zenitude Internship

---
## 🚀 Featured Project — WorkPulse

**Company Culture Sentiment Analyzer** | NLP + Full-Stack | [Live Demo →](https://workpulse-frontend-pied.vercel.app)

Fine-tuned DistilBERT on 838K real Glassdoor reviews to classify company culture as Positive, Negative, Neutral, or Mixed.
| | |
|---|---|
| 🌐 Live Demo | https://workpulse-frontend-pied.vercel.app |
| 🔌 API | https://madhuri1003-workpulse-api.hf.space/docs 
|
| 🤗 Model | https://huggingface.co/Madhuri1003/workpulse-distilbert |

**Tech:** DistilBERT · FastAPI · React · HuggingFace Spaces · Vercel · 838K Glassdoor Reviews

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
| Week 4 | Deep Learning Fundamentals |✅ Complete  | CIFAR-10 classifier, Andrew Ng Course 1 |
| Week 5 | NLP & Advanced DL | ⏳ Upcoming | |
| Week 6 | Full-Stack for ML |  | |
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

### WorkPulse — Company Culture Sentiment Analyzer
**Weeks 6-7 | NLP + Full-Stack | [Live Demo](https://workpulse-frontend-pied.vercel.app)**

Fine-tuned DistilBERT on 838K real Glassdoor reviews for 3-class sentiment classification.

- Dataset: 838K Glassdoor reviews with per-aspect ratings
- Key insight: 46.2% of negative reviews had positive language in pros field — solved via deliberate label-text mapping
- Mixed sentiment detection using contrast keyword splitting
- Full-stack deployment: HuggingFace Spaces (backend) + Vercel (frontend)
- Model accuracy: 78.9% | F1 Macro: 0.789

[View Backend →](./Week5-NLP-WorkPulse/) | [View Frontend →](./Week7-FullStack-WorkPulse/)

---


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
## Skills Covered

**NLP & Deep Learning**
DistilBERT fine-tuning, HuggingFace Transformers, Sentiment Analysis, Text Classification

**Full-Stack ML**
FastAPI, React, REST APIs, CORS, Docker, HuggingFace Spaces, Vercel

**Supervised Learning**
Linear Regression, Ridge, Lasso, Logistic Regression, Random Forest, XGBoost, SVM

**Unsupervised Learning**
K-Means Clustering, PCA, DBSCAN

**Model Evaluation**
Cross-validation, GridSearchCV, Confusion Matrix, F1 Score, ROC-AUC, RMSE, R²

**Data & Tools**
Python, Pandas, NumPy, Scikit-learn, PyTorch, Matplotlib, Seaborn, Jupyter, Git

**Other**
SQL (HackerRank Intermediate Certificate), EDA, Feature Engineering, Kaggle Competitions
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

### Weeks 5-7 — NLP & Full-Stack ML
Built WorkPulse end-to-end: fine-tuned DistilBERT on 838K Glassdoor reviews, built FastAPI backend, React frontend with mixed sentiment detection, deployed full stack live.

[View Backend →](./Week5-NLP-WorkPulse/) | [View Frontend →](./Week7-FullStack-WorkPulse/)
