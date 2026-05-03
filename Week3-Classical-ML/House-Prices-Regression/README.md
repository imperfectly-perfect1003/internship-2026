# House Prices — Advanced Regression Techniques
### Kaggle Competition | My Approach and Thought Process

---

## What is this competition about?

The goal is simple — predict the sale price of residential homes in Ames, Iowa based on 79 different features describing almost every aspect of the house. Unlike Titanic where you're predicting 0 or 1 (survived or not), this is a **regression problem** — you're predicting an actual number (the price).

The evaluation metric Kaggle uses here is **RMSLE** — Root Mean Squared Log Error. The reason they use log error instead of regular error is interesting: a $10,000 mistake on a $50,000 house is much worse proportionally than a $10,000 mistake on a $500,000 house. Log error penalizes both equally in relative terms, which is fairer.

---

## Understanding the Dataset

The training data has **1460 houses and 81 columns**. The test data has 1459 houses and 80 columns (no SalePrice — that's what we predict).

79 features is a lot. I spent time grouping them mentally into categories before doing anything else, because with that many features you can easily get lost.

### The 3 Things That Determine a House Price

After reading through all 79 features I realized every feature falls into one of three buckets:

**1. Structure — what's actually built**
This is the physical stuff. Square footage, number of rooms, basement size, garage size, number of bathrooms. These are the features that tell you how big and well-equipped the house is. Key ones: GrLivArea, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GarageArea, FullBath, BedroomAbvGr.

**2. Location — where it sits**
Neighborhood, zoning, lot size, lot shape, proximity to roads. Two identical houses in different neighborhoods can have very different prices. NridgHt neighborhood had a median price of ~$315,000 while MeadowV was ~$88,000 — that's a 3.5x difference just from location.

**3. Condition — how well kept it is**
Overall quality rating (1-10), overall condition (1-10), year built, year remodeled, exterior quality, kitchen quality, basement quality. A well-maintained older house can be worth more than a neglected newer one.

This "Value Decomposition" thinking guided all my feature engineering decisions later.

---

### Key Columns Explained

**SalePrice** — the target. Ranges from $34,900 to $755,000 with a mean of ~$180,000. Heavily right-skewed (most houses are affordable, a few are very expensive).

**OverallQual** — single most important feature. Rated 1-10 by the assessor. A rating of 10 means "Very Excellent" quality materials and finish. Correlation with SalePrice is 0.79 — the highest of any single feature.

**GrLivArea** — above ground living area in square feet. Second most correlated with price (0.71). Makes total sense — bigger house = more expensive.

**GarageCars / GarageArea** — number of cars the garage fits and its area. Both around 0.62-0.64 correlation. Garage quality is a big deal in American suburbs.

**TotalBsmtSF** — total basement square footage (0.61 correlation). Finished basements add significant value.

**YearBuilt / YearRemodAdd** — when the house was built and last remodeled. Newer houses and recently renovated ones command higher prices (~0.52 correlation each).

**Neighborhood** — categorical but crucial. 28 unique neighborhoods with massive price differences. NridgHt, NoRidge, StoneBr are the most expensive. MeadowV, IDOTRR, BrDale are the cheapest.

**PoolQC, MiscFeature, Alley, Fence** — over 80-99% missing. These aren't really "missing" — most houses just don't have pools, alleys, or special fencing. The absence itself is the information.

**FireplaceQu** — 47% missing. Again, about half the houses just don't have a fireplace.

**LotFrontage** — 17.74% missing. This one is genuinely missing (not "no lot frontage") — I handled it by filling with the median of the same neighborhood, which is a smarter approach than just using the global median.

---

## Step 1 — Exploratory Data Analysis

### SalePrice Distribution — Why Log Transform?

The first thing I plotted was the SalePrice distribution. Original skewness was **1.88** — heavily right-skewed. A handful of very expensive houses were pulling the distribution to the right.

Log transforming brought the skewness down to **0.12** — almost perfectly normal. This matters for two reasons:

1. Linear regression assumes the target is normally distributed. Skewed target = worse linear model performance.
2. The competition uses RMSLE anyway — predicting log(price) and then converting back is exactly what RMSLE measures. So we're optimizing for the right thing.

I used `np.log1p()` (log of 1+x) instead of regular log because it handles zeros gracefully, though SalePrice doesn't have zeros anyway.

### Missing Values

19 columns had missing values. The important insight here was that most missing values **weren't really missing** — they just meant the house didn't have that feature:

- PoolQC missing → house has no pool
- FireplaceQu missing → house has no fireplace
- GarageType missing → house has no garage
- Alley missing → house has no alley access

Treating these as "None" or 0 instead of trying to impute them was the right call. Imputing a pool quality rating for a house that doesn't have a pool would be completely wrong.

### Top Correlations

The top features correlated with SalePrice (from the correlation bar chart):
- OverallQual: 0.79
- GrLivArea: 0.71
- GarageCars: 0.64
- GarageArea: 0.62
- TotalBsmtSF: 0.61

One thing I noticed from the correlation heatmap — GarageCars and GarageArea have 0.88 correlation with each other. They're basically measuring the same thing (bigger garage = fits more cars). This multicollinearity is fine for tree models but can hurt linear regression — which is exactly why Ridge and Lasso (which handle multicollinearity better than plain linear regression) performed so much better.

### Key Scatter Plots

**OverallQual vs SalePrice**: Clean staircase pattern. Each quality level has a higher median price than the one below it, and the gap gets bigger as quality increases. Very non-linear relationship.

**GrLivArea vs SalePrice**: Strong linear trend but with two obvious outliers — very large houses (>4000 sqft) with suspiciously low prices. These are likely partial sales or special circumstances. I removed them because they would confuse the model.

**YearBuilt vs SalePrice**: Newer houses are generally more expensive but the relationship isn't perfectly linear. Houses from before 1940 show more variance — some are well-maintained historical properties, some are run-down.

### Neighborhood Analysis

The neighborhood bar chart was one of the most revealing plots. NridgHt (~$315K median) vs MeadowV (~$88K median) — same city, completely different price range. Location really is everything. This reinforced that neighborhood needed to be properly encoded, not dropped.

---

## Step 2 — Data Cleaning

### Why I Combined Train and Test Before Cleaning

I combined train and test data into one dataframe (2917 rows total) before cleaning. The reason: if I cleaned them separately, I might end up with different columns after one-hot encoding (train might have a category that test doesn't, or vice versa). Cleaning together guarantees both datasets have identical columns at the end.

I saved the SalePrice separately before combining and removed it from the combined dataframe.

### Three Types of Missing Value Fixes

**Type 1 — Fill with "None" (no feature exists)**
Columns like PoolQC, GarageType, FireplaceQu, BsmtQual etc. Missing means the feature doesn't exist. Filled with the string "None" so the model knows this is a valid category.

**Type 2 — Fill with 0 (numeric, no feature exists)**
Columns like GarageArea, GarageYrBlt, BsmtFinSF1, MasVnrArea. Same logic but numeric — if there's no garage, garage area is 0.

**Type 3 — Fill with mode**
Columns like MSZoning, Electrical, KitchenQual — these are genuinely missing, not "no feature". Filled with most common value. Only a handful of rows affected.

**LotFrontage — smarter imputation**
Instead of filling with the global median, I filled LotFrontage with the median of the same neighborhood. Houses in the same neighborhood tend to have similar lot sizes. This is more accurate than a one-size-fits-all median fill.

I also dropped the Utilities column entirely — almost all houses had the same value (AllPub) so it would add noise without any signal.

---

## Step 3 — Feature Engineering

This is the step I'm most proud of. Instead of just using the 79 raw features, I created 10 new ones based on my understanding of what actually drives house prices.

### TotalSF — Total Square Footage
```
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
```
The dataset splits square footage across three columns. A buyer doesn't care which floor the space is on — they care about total livable area. This single feature ended up being the third most important in the XGBoost model.

### HouseAge and RemodelAge
```
HouseAge = YrSold - YearBuilt
RemodelAge = YrSold - YearRemodAdd
```
Instead of raw year built (which is an absolute number), age gives the model relative information — how old was the house when it was sold? A house built in 1990 sold in 2010 is 20 years old. That's more meaningful than the year 1990 itself.

### TotalBath — Combined Bathroom Count
```
TotalBath = FullBath + BsmtFullBath + 0.5*HalfBath + 0.5*BsmtHalfBath
```
Four separate bathroom columns combined into one. Half bathrooms (no shower/tub) count as 0.5. This is how real estate agents actually think about bathrooms — "3.5 bath house" means 3 full and 1 half.

### Binary Presence Features
HasPool, HasGarage, HasBsmt, HasFireplace — simple 0/1 flags. The presence or absence of these amenities matters even before considering their size or quality.

### TotalPorch
Combined all porch types (open, enclosed, 3-season, screened) into one. Same logic as TotalSF — buyers think about total outdoor space, not which type of porch it is.

### QualCond — Quality × Condition Interaction
```
QualCond = OverallQual × OverallCond
```
A house with high quality (9) but poor condition (3) is worth less than a house with decent quality (7) and great condition (8). Multiplying them captures this interaction. The model can't easily figure this out on its own from two separate columns.

### QualLivArea — Quality × Living Area Interaction
```
QualLivArea = OverallQual × GrLivArea
```
This ended up being the **single most important feature** in the XGBoost model. The insight: a large house with high quality materials is worth disproportionately more than a large house with average materials. Size and quality don't just add — they multiply.

### Encoding Categorical Features

For ordinal quality columns (ExterQual, KitchenQual, BsmtQual etc.) I used a manual quality map:
None=0, Po=1, Fa=2, TA=3, Gd=4, Ex=5

This preserves the natural ordering — Excellent really is better than Good which is better than Fair. Label encoding random numbers would lose this information.

For all remaining categorical columns I used one-hot encoding (get_dummies), which went from 89 columns to 270 columns. Each category becomes its own binary column.

### Fixing Skewed Numerical Features

236 numerical features had skewness above 0.75. Applied log1p transform to all of them. Same reasoning as the SalePrice transform — skewed features hurt linear models. Tree-based models handle skewness fine, but since we're using Ridge and Lasso too, fixing skewness helped those models significantly.

---

## Step 4 — Model Building

### Why RobustScaler?

Before training linear models I scaled the features using RobustScaler instead of the more common StandardScaler. RobustScaler uses median and IQR instead of mean and standard deviation, which makes it resistant to outliers. Since house price data has many outliers in features like GarageArea and LotArea, RobustScaler is a better choice here.

Tree-based models (Random Forest, XGBoost) don't need scaling since they split on feature thresholds anyway. I used scaled data for linear models and unscaled for tree models.

### Linear Regression (Baseline)
RMSE: 0.2616 — terrible. With 270 features and only 1458 training samples, plain linear regression is massively overfitting. It memorizes the training data but fails to generalize. The huge standard deviation (0.2582) confirms this — the model is unstable across different CV folds.

This is exactly why regularization exists.

### Ridge Regression (L2 Regularization)
RMSE: 0.1135

Ridge adds a penalty to the loss function that discourages large coefficients:
```
Loss = RSS + α × sum(coefficients²)
```
The α=10 penalty forces the model to keep coefficients small, preventing any single feature from dominating. This dramatically reduces overfitting and brought the RMSE from 0.26 down to 0.11. The low standard deviation (0.0075) shows it's very stable across folds.

Ridge keeps all features but shrinks their coefficients. It's particularly good when many features are mildly useful (which is the case here — most of the 270 features contribute something).

### Lasso Regression (L1 Regularization)
RMSE: 0.1126 — best single model

Lasso is similar to Ridge but uses absolute values instead of squares:
```
Loss = RSS + α × sum(|coefficients|)
```
The key difference: Lasso can shrink coefficients all the way to zero, effectively doing automatic feature selection. With 270 features, many of which are redundant one-hot encoded categories, Lasso's ability to zero out useless features gave it a slight edge over Ridge.

The α=0.0005 is very small — we don't want to be too aggressive with elimination since most features do contribute something.

### Random Forest Regressor
RMSE: 0.1312 — surprisingly worse than linear models here

Random Forest usually beats linear models but not in this case. The reason is likely the very high dimensionality (270 features) — when features outnumber meaningful signals by that much, random subsampling of features at each split (which RF does) can miss the important ones. Also RF tends to struggle with extrapolation on house prices — it can't predict prices higher than the max in training data.

### XGBoost Regressor
RMSE: 0.1139 — close to Ridge and Lasso

XGBoost does better than Random Forest because the boosting approach (each tree correcting previous errors) is more efficient with high-dimensional data. The n_estimators=2000 with learning_rate=0.05 is deliberately slow and careful — many small steps rather than fewer large ones.

### Stacking Ensemble
RMSE: 0.1285

Interestingly the stacking ensemble didn't outperform the individual linear models. This sometimes happens when the base models are already well-tuned and the meta-model doesn't find much to improve on. The Ridge and Lasso were already doing an excellent job of capturing the signal.

### Final Blending Strategy
Instead of picking one model, I blended the three best:
```
Final = Ridge(25%) + Lasso(35%) + XGBoost(40%)
```
Lasso gets the highest weight since it was the best individual model. XGBoost gets the second highest because it captures non-linear patterns that linear models miss. Ridge fills in the gaps.

---

## Step 5 — Evaluation

### R² Score: 0.923

R² of 0.923 means the model explains 92.3% of the variance in house prices. In plain terms — the model's predictions are very close to reality 92% of the time. For a dataset with 79 noisy features about real estate, this is a strong result.

### Residual Analysis

The residuals vs predicted plot showed residuals scattered randomly around 0 with no obvious pattern — which is exactly what you want. If there was a pattern (like residuals increasing with price), it would mean the model is systematically wrong in certain price ranges.

The residual distribution was roughly normal, slightly left-skewed. A few predictions were significantly off (residuals around -0.6 to -0.8) — these are likely unusual properties that don't fit the normal patterns (unique architecture, special circumstances etc.).

### Feature Importance (XGBoost Top 20)

The most important features confirmed the feature engineering worked:

1. **QualLivArea** — the interaction feature I created. Most important by far.
2. **ExterQual** — exterior quality rating. Makes sense, first impression matters.
3. **TotalSF** — the combined square footage feature I created. Third most important.
4. **OverallQual** — the overall quality rating.
5. **GarageCars** — garage capacity.

Two of the top three features were ones I engineered myself (QualLivArea and TotalSF). This confirms that domain-informed feature engineering is more valuable than just throwing raw features at the model.

---

## Step 6 — Predictions and Submission

Applied the identical pipeline to test data:
- Same missing value treatment
- Same feature engineering
- Same encoding
- Same scaling

Generated predictions using the blended model and reversed the log transform with `np.expm1()` to get actual dollar prices.

Final predictions:
- Min: $43,248
- Max: $812,431
- Mean: $178,323

These ranges are realistic for Ames, Iowa housing market. The mean of $178K is close to the training data mean of ~$180K, which is a good sign the model isn't systematically over or under predicting.

---

## Results Summary

| Model | CV RMSE | Notes |
|-------|---------|-------|
| Linear Regression | 0.2616 | Massively overfit — 270 features, too many |
| Ridge | 0.1135 | L2 regularization fixes overfitting |
| Lasso | 0.1126 | Best single model, L1 does feature selection |
| Random Forest | 0.1312 | Struggles with high dimensionality |
| XGBoost | 0.1139 | Captures non-linear patterns well |
| Stacking Ensemble | 0.1285 | Didn't improve over base models |
| **Final Blend** | **0.1139** | Ridge 25% + Lasso 35% + XGBoost 40% |

**Validation R² Score: 0.923**

---

## Key Lessons from This Project

**1. Log transforming the target was the single most impactful preprocessing step.** SalePrice skewness of 1.88 was hurting every model. Bringing it to 0.12 made a massive difference, especially for the linear models.

**2. Feature engineering beats raw features.** QualLivArea (my engineered feature) was more important than any raw column in the dataset. Understanding what drives prices — quality multiplied by size — gave the model information it couldn't derive from the raw columns alone.

**3. Regularization is essential with high-dimensional data.** Plain linear regression got 0.26 RMSE. Ridge got 0.11. The only difference was the regularization penalty. With 270 features, you need regularization — there's no way around it.

**4. Missing values aren't always missing.** Most "missing" values in this dataset meant "this feature doesn't exist." Treating PoolQC missing as "no pool" rather than "unknown pool quality" is a fundamentally different and correct interpretation.

**5. Smarter imputation beats simple imputation.** Filling LotFrontage with neighborhood median instead of global median is a small thing but it's the kind of careful thinking that separates good data science from mediocre data science.

