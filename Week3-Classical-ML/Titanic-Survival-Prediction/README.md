# Titanic Survival Prediction — Kaggle Competition
### My Approach, Thought Process and Findings

---

## What is this competition about?

The Titanic competition on Kaggle is one of the most well-known beginner ML competitions. The goal is simple — given information about passengers on the Titanic, predict whether they survived or not. It's a binary classification problem (survived = 1, did not survive = 0).

The reason this dataset is so popular for learning is that the patterns in it actually make real-world sense. It's not just random numbers — every feature connects back to something that actually happened on the night the ship sank. That makes it way easier to reason about what features should matter and why.

---

## Understanding the Dataset

Before writing a single line of code I spent time just understanding what each column actually means, because if you don't know your data you're just blindly throwing features at a model.

The dataset has 891 rows (passengers) and 12 columns:

**PassengerId** — just a serial number. No predictive value at all, only needed for the submission file.

**Survived** — the target variable. 0 means the passenger died, 1 means they survived. About 38% of passengers survived, so the dataset is somewhat imbalanced.

**Pclass** — passenger class. 1 = first class (wealthy), 2 = second class, 3 = third class (cheapest). This turned out to be one of the strongest predictors. First class cabins were on upper decks, physically closer to the lifeboats.

**Name** — full name including title (Mr, Mrs, Miss, Master etc). Looks useless at first but the title hidden inside it is actually very valuable — more on this in the feature engineering section.

**Sex** — male or female. This ended up being the single most important feature. The "women and children first" evacuation policy shows up very clearly in the data. Female survival rate was ~74%, male was ~19%.

**Age** — age in years, ranging from 0.42 (an infant) to 80. About 20% of values are missing which needed to be handled. Age matters somewhat — children had better odds than adult men — but it's not the strongest predictor on its own.

**SibSp** — number of siblings or spouse travelling with the passenger. A value of 0 means they had no spouse or siblings on board.

**Parch** — number of parents or children travelling with the passenger. Combined with SibSp, this tells you whether someone was travelling alone or with family.

**Ticket** — ticket number. Mostly a mix of random strings and numbers. Some passengers share ticket numbers (travelling together) but it's hard to extract anything useful from this without a lot of extra work. I dropped it.

**Fare** — how much the passenger paid for their ticket, in British pounds. Ranges from £0 to £512. Strongly correlated with Pclass — first class passengers paid more. Higher fare generally meant better survival odds.

**Cabin** — cabin number like C85 or B38. 77% of this column is missing, which itself is informative — most 3rd class passengers didn't have a cabin record. I converted this to a binary feature (has cabin record or not) instead of using the raw values.

**Embarked** — port where the passenger boarded. S = Southampton (72%), C = Cherbourg (19%), Q = Queenstown (9%). Passengers who boarded at Cherbourg had slightly better survival rates, likely because more 1st class passengers boarded there.

---

## The Story Behind the Data

When you look at the data through the lens of what actually happened that night, everything makes sense:

1. **"Women and children first"** — this evacuation policy was actually followed, which is why Sex is the strongest predictor
2. **First class cabins were on upper decks** — physically closer to lifeboats, explains why Pclass matters so much
3. **Wealthy passengers had more information and time** — explains why Fare correlates with survival
4. **Solo travellers had nobody looking out for them** — explains why travelling alone hurt your odds
5. **Large families couldn't move quickly together** — explains why very large family sizes also had worse outcomes

Understanding this context helped me decide which features to create and which to drop.

---

## Step 1 — Exploratory Data Analysis

The first thing I did was just look at the data from every angle before touching it. EDA is about building intuition — what does the data look like, what's missing, what correlates with survival.

**Missing values** were the first thing I checked. Three columns had missing data:
- Cabin: 77.1% missing — way too much to use directly
- Age: 19.87% missing — significant but manageable
- Embarked: 0.22% missing — basically nothing, easy to fix

**Survival distribution** showed that only about 342 out of 891 passengers survived (~38%). This class imbalance is something to keep in mind when evaluating model performance — more on this in the evaluation section.

**Age distribution** was right-skewed with most passengers between 20-35 years old. There was a noticeable cluster of very young passengers near age 0-5.

**Passenger class distribution** showed that most passengers (~490) were in 3rd class, followed by 1st class (~216) and 2nd class (~184).

The most revealing plots were the bivariate ones:

- **Survival by gender**: Female passengers survived at a dramatically higher rate. About 468 males died vs only 109 survived. For females, 233 survived vs only 81 died. Clearest pattern in the entire dataset.

- **Survival by class**: 1st class had more survivors than deaths. 3rd class had massively more deaths (~372 deaths vs only 119 survivors). Wealth quite literally determined your survival odds.

- **Fare vs survival**: Survivors paid noticeably higher fares on average — directly tied to the class effect.

- **Correlation heatmap**: Pclass had -0.34 correlation with survival, Fare had +0.26. Sex wasn't in the numeric heatmap but was clearly the strongest predictor from the bar charts.

I also checked survival rates by embarked port and family size which fed into my feature engineering decisions.

---

## Step 2 — Data Cleaning

After EDA I knew exactly what needed to be fixed. I always work on a copy of the original dataframe so if something goes wrong I don't have to reload everything.

**Filling Age missing values with median**: Chose median over mean because Age has some older outliers that would pull the mean upward, giving unrealistic fill values. Median (28.0) is more robust to outliers.

**Filling Embarked with mode**: Only 2 values missing, filled with 'S' (Southampton) since ~72% of passengers boarded there.

**Cabin → HasCabin binary feature**: Instead of dropping Cabin entirely, I first extracted whether a passenger had a cabin record or not. Having a cabin record likely means first or second class — so even though the actual cabin number is mostly missing, the presence or absence of a record still carries useful information.

**Dropped PassengerId, Name, Ticket**: Serial number, random strings, no predictive value. Name was dropped after extracting Title from it in feature engineering.

**Encoded Sex and Embarked**: Male → 0, Female → 1. S → 0, C → 1, Q → 2. Models only understand numbers. For Sex this is perfectly fine since it's binary. For Embarked, label encoding technically implies an ordering but for tree-based models this doesn't matter since they split on thresholds anyway.

After cleaning, every column had 0 missing values and all dtypes were numeric.

---

## Step 3 — Feature Engineering

This was the most impactful step for the model score. Raw columns don't always tell the full story — sometimes creating new features captures patterns more directly.

**Title extraction from Name**: Every passenger name includes a title. I extracted this using regex and grouped rare titles (Dr, Rev, Col, Major, Countess etc.) into a single "Rare" category.

Why is this so useful? Title captures more than just gender:
- Master specifically identifies young boys (under ~13) who had better odds than adult men
- Mrs vs Miss distinguishes married vs unmarried women
- Without Title, the model treats all males the same — but a 5 year old boy and a 40 year old man had very different survival odds

Survival rates confirmed this: Mrs → 79.2%, Miss → 69.78%, Master → 57.5%, Rare → 34.78%, Mr → 15.67%.

**FamilySize and IsAlone**: Combined SibSp and Parch into FamilySize (SibSp + Parch + 1, the +1 includes the passenger themselves). Then created binary IsAlone (FamilySize == 1). 537 out of 891 passengers were travelling completely alone.

Solo travellers had worse odds, small families (2-4) had the best odds, and very large families (5+) also struggled because they couldn't move quickly together.

**AgeBand**: Binned Age into 5 groups — Child (0-12), Teen (13-18), Young Adult (19-35), Middle Aged (36-60), Senior (61-80). Survival doesn't change linearly with age — the difference between a child and an adult is more meaningful than between a 20 and 30 year old. Binning captures these group-level effects better.

**FareBand**: Split Fare into 4 quartile-based groups. Similar reasoning — fare's relationship with survival is non-linear.

---

## Step 4 — Model Building and Tuning

I trained three different models, compared them, tuned the best ones, and combined them into an ensemble.

**Logistic Regression** — simplest baseline. Draws a straight line to separate survivors from non-survivors. Good for linear relationships but can't handle complex interactions like "female AND 3rd class AND alone" together. Ended up at ~79% accuracy as expected.

**Random Forest** — builds hundreds of decision trees and takes a majority vote. Handles non-linear patterns much better. Used max_depth=6 to prevent overfitting — unlimited depth trees memorize training data but fail on new data.

**XGBoost** — builds trees sequentially where each new tree corrects the mistakes of the previous one. Consistently the best single model in cross-validation at 82.27% mean accuracy.

**Cross Validation**: Used 5-fold CV instead of a single train/test split. Tests on 5 different splits and averages — gives a much more reliable accuracy estimate. Results: LR 81.48% (std 1.47%), RF 81.71% (std 2.99%), XGB 82.27% (std 2.85%).

**Hyperparameter Tuning**: GridSearchCV tested 405 combinations for RF and 270 for XGBoost across 5 folds each. Best RF params: max_depth=6, min_samples_split=10, n_estimators=200 → CV score 83.70%. Best XGB params: learning_rate=0.05, max_depth=4, n_estimators=200, subsample=0.8 → CV score 83.14%.

**Voting Ensemble**: Combined all three using soft voting — each model gives a probability and the average determines the final prediction. Works because each model makes different mistakes and combining them cancels individual errors. Final accuracy: 83.24%.

---

## Step 5 — Model Evaluation

Accuracy alone is misleading on imbalanced datasets. A model that always predicts "died" would get 62% accuracy without learning anything. So I evaluated using four metrics:

### Confusion Matrix Results

Out of 179 test passengers:
- 91 correctly predicted as died (True Negatives)
- 58 correctly predicted as survived (True Positives)
- 14 wrongly said survived but actually died (False Positives)
- 16 wrongly said died but actually survived (False Negatives)

The model misses 16 actual survivors — these are the worst errors because those are real people the model failed to identify correctly. The false positives (14) are slightly fewer.

### Full Metrics — Voting Ensemble

| Metric | Score |
|--------|-------|
| Accuracy | 83.24% |
| F1 Score | 79.45% |
| Precision | 80.56% |
| Recall | 78.38% |

Precision of 80.56% means when the model predicts someone survived, it's right about 80% of the time. Recall of 78.38% means it correctly identifies about 78% of all actual survivors. F1 being slightly lower than accuracy confirms the dataset imbalance was flattering the accuracy number a little.

### ROC-AUC Results

| Model | AUC |
|-------|-----|
| Voting Ensemble | 0.903 |
| XGBoost | 0.895 |
| Logistic Regression | 0.894 |
| Random Forest | 0.892 |

AUC of 0.903 is a strong result. It means if you randomly picked one survivor and one non-survivor from the dataset, the model would correctly rank the survivor as "more likely to survive" 90.3% of the time. Anything above 0.85 is generally considered very good.

Interesting observation — Logistic Regression scores 0.894 AUC despite only 79% accuracy. This means it's actually ranking passengers well by probability — it just uses a slightly worse threshold to convert those probabilities into hard 0/1 predictions. AUC is a better measure of the model's underlying discrimination ability.

All four models scored above 0.89 AUC, which means even the weakest model is doing genuinely useful work — not just getting lucky.

---

## Step 6 — Test Data and Submission

Applied the exact same cleaning and feature engineering pipeline to the test data (418 rows). The most important detail: used statistics from the training data (train median for Age, train mode for Embarked) when filling test missing values — not test statistics. In a real scenario you wouldn't have the test data during training so you should only use what training data tells you.

Generated predictions using the Voting Ensemble and saved to submission.csv.


---

## Key Takeaways

The single most important lesson from this project is that **feature engineering matters more than model choice**. Going from raw features to engineered features — especially Title extraction — had a bigger impact on accuracy than switching between models. You can't do good feature engineering without understanding the domain first.

The second lesson is that **accuracy alone is not enough**. The ROC-AUC of 0.903 and the confusion matrix breakdown told a much more complete story than the 83% accuracy number alone. In a real medical or financial application, knowing exactly what kind of errors your model makes matters a lot more than the overall accuracy.

---

