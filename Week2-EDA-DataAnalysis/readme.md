# Week 2 — Data Analysis & Visualization

## Deliverables Status

| Deliverable | Status |
|-------------|--------|
| Complete Kaggle EDA notebook on a real dataset | ✅ Completed |
| HackerRank SQL Intermediate Badge/Certificate | ✅ Completed |
| Push EDA project to GitHub | ✅ Completed |

---

## Courses & Certificates Completed

- ✅ **HackerRank SQL Intermediate Certificate** — earned this week
- ✅ **EDA on Titanic dataset** — full exploratory analysis notebook

---

## SQL — HackerRank Intermediate Certificate

Completed the HackerRank SQL Intermediate badge covering:
- Window functions (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD)
- Common Table Expressions (CTEs)
- Subqueries and correlated subqueries
- Complex multi-table joins (INNER, LEFT, RIGHT, FULL OUTER)
- Aggregations with HAVING
- CASE statements and conditional logic

> Certificate earned on HackerRank — SQL Intermediate level

---

## EDA — Titanic Dataset

**Note:** The full EDA notebook is part of the Week 3 Titanic project
since the EDA directly fed into the model building pipeline.
The EDA is the first half of the Titanic notebook —
Steps 1 through the correlation heatmap.

→ [View Full EDA + Titanic Project](../Week3-Classical-ML/Titanic-Survival-Prediction/)

### What the EDA Covered

**Missing Value Analysis**
- Cabin: 77.1% missing — converted to HasCabin binary feature
- Age: 19.87% missing — filled with median
- Embarked: 0.22% missing — filled with mode
- Visualized with heatmap to show pattern of missingness

**Univariate Analysis**
- Survival count distribution (62% died, 38% survived)
- Age distribution — right skewed, peak at 20-35
- Passenger class distribution — majority in 3rd class (~490)

**Bivariate Analysis**
- Survival by gender — females 74% survival, males 19%
- Survival by class — 1st class 63%, 2nd 47%, 3rd 24%
- Age vs Survival boxplot — similar medians, weak effect
- Fare vs Survival boxplot — survivors paid higher fares

**Survival Rate Analysis**
- Grouped survival rates by Sex, Pclass, Embarked port
- Embarked C (Cherbourg) had best survival — more 1st class boarded there

**Correlation Heatmap**
- Pclass: -0.34 with Survived
- Fare: +0.26 with Survived
- Pclass and Fare: -0.55 with each other (multicollinearity noted)

**Family Size Effect**
- Created FamilySize = SibSp + Parch + 1
- Solo travellers had worse survival odds
- Small families (2-4) had best survival odds
- Very large families also suffered — couldn't move quickly together

**Plotly Interactive Visualization**
- Age distribution by Passenger Class and Survival (faceted histogram)
- Showed Class 3 young adults had worst outcomes despite being largest group

### Tools Used
- Pandas for data manipulation
- Matplotlib and Seaborn for static plots
- Plotly Express for interactive visualization

### Key EDA Findings
1. Sex is the strongest predictor — "women and children first" shows in the data
2. Pclass is second strongest — cabin deck location determined lifeboat access
3. Fare correlates with survival (tied to class)
4. Age has weak effect individually but strong effect combined with other features
5. Cabin being missing (77%) is itself informative — most 3rd class had no record