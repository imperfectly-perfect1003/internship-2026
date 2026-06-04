# Mall Customer Segmentation — Unsupervised Learning
### K-Means | PCA | DBSCAN | SVM | RandomizedSearchCV

---

## What is this project about?

This notebook covers unsupervised machine learning on the Mall Customer Segmentation dataset. Unlike Titanic or House Prices where I had a target variable to predict, unsupervised learning means there's no "right answer" given to the model. The goal is to find hidden patterns and structure in the data on its own.

The business question here is: **can we group mall customers into meaningful segments based on their age, income, and spending behaviour?** If yes, the mall can use those segments to target marketing — different campaigns for different types of customers.

This notebook also covers SVM classification and RandomizedSearchCV tuning, which completes the supervised learning syllabus requirements alongside the unsupervised techniques.

---

## Understanding the Dataset

200 mall customers, 5 columns:

**CustomerID** — just a serial number, dropped before any analysis.

**Genre (Gender)** — male or female. The dataset uses "Genre" which is an unusual naming choice — renamed to "Gender" for clarity. 112 females (56%) and 88 males (44%) — slightly more female customers.

**Age** — ranges from 18 to 70, mean of ~39 years. Distribution is right-skewed — more younger customers than older ones. Peak around 30-35 years.

**Annual Income (k$)** — annual income in thousands of dollars. Ranges from $15k to $137k, mean of ~$60k. Roughly bell-shaped distribution centered around $60-70k.

**Spending Score (1-100)** — this is the most interesting column. It's a score assigned by the mall based on customer behaviour and purchase data. 1 = very low spender, 100 = very high spender. Distribution is fairly uniform — customers are spread across all spending levels.

### Key Observations Before Any Modelling

From the correlation heatmap, Age and SpendingScore have a -0.33 correlation — older customers tend to spend less. Income and SpendingScore have almost zero correlation (0.01) which is surprising — having more money doesn't mean you spend more at the mall. This already hints that income and spending are independent dimensions, which is exactly why the clustering turns out the way it does.

The Income vs SpendingScore scatter plot tells the whole story before any algorithm runs — you can visually see 5 distinct groups just by looking at it. This is actually a relatively easy clustering problem, which explains why K-Means performs so well.

---

## Step 1 — Exploratory Data Analysis

Standard EDA first — distributions, gender split, pairplot, correlations.

The pairplot coloured by gender showed that male and female customers have very similar distributions across all three features. Gender doesn't create meaningful separation in the data. This told me gender shouldn't be a primary clustering feature — I used Income and SpendingScore as the main clustering dimensions instead.

The Income vs SpendingScore scatter plot was the most informative — even before running any algorithm you can clearly see 5 groups:
- Bottom left: low income, low spending
- Top left: low income, high spending  
- Middle: average income, average spending (the big dense cluster)
- Top right: high income, high spending
- Bottom right: high income, low spending

This visual intuition guided the K-Means analysis and helped validate that K=5 was the right choice before even running the elbow method.

---

## Step 2 — K-Means Clustering

### Why K-Means?

K-Means is the right starting point for customer segmentation because:
- We expect roughly spherical, similarly-sized clusters (which is what the scatter plot suggested)
- It's interpretable — each customer belongs to exactly one segment
- It scales well and is fast

The only problem with K-Means is you have to tell it how many clusters to use. That's what the Elbow Method and Silhouette Score are for.

### Finding Optimal K — Elbow Method + Silhouette Score

I used both methods together because neither is perfect alone:

**Elbow Method** — plots inertia (sum of distances from each point to its cluster center) against K. The "elbow" where the curve bends is the optimal K. From my plot the elbow is clearly at K=5 — after 5 the inertia reduction slows down significantly.

**Silhouette Score** — measures how well each point fits its own cluster compared to other clusters. Ranges from -1 to 1, higher is better. My silhouette scores peaked at K=5 (0.5547) and dropped after that. Both methods agreed on K=5 which gave me confidence.

A silhouette score of 0.5547 is solid for customer segmentation — it means clusters are reasonably well separated and compact.

### The 5 Customer Segments

After running K-Means with K=5, the cluster profiles were:

| Cluster | Avg Age | Avg Income | Avg Spending | Count | Segment Label |
|---------|---------|------------|--------------|-------|---------------|
| 0 | 42.7 | $55.3k | 49.5 | 81 | Regular Customers |
| 1 | 32.7 | $86.5k | 82.1 | 39 | Target Customers |
| 2 | 25.3 | $25.7k | 79.4 | 22 | Impulsive Buyers |
| 3 | 41.1 | $88.2k | 17.1 | 35 | Careful Spenders |
| 4 | 45.2 | $26.3k | 20.9 | 23 | Budget Customers |

**Regular Customers (81 people)** — the biggest group. Middle income, middle spending. Average everything. The mall's bread and butter — most reliable but hardest to move up or down.

**Target Customers (39 people)** — high income AND high spending. These are the dream customers. Young (avg 32.7), wealthy, and willing to spend. Every mall's marketing priority.

**Impulsive Buyers (22 people)** — low income but high spending. Young (avg 25.3). They don't earn much but they spend a lot anyway — probably on trends and experiences. Marketing to them with deals and limited offers could work well.

**Careful Spenders (35 people)** — high income, low spending. Older (avg 41.1), earn well but are conservative with money. Hard to convert but high potential if you can win their trust.

**Budget Customers (23 people)** — low income, low spending. Older (avg 45.2). Limited room to grow spending. Marketing spend here has low ROI.

These labels make intuitive business sense which is a good sign the clustering is working correctly.

---

## Step 3 — PCA (Principal Component Analysis)

### Why PCA?

PCA serves two purposes here:

1. **Dimensionality reduction** — compress multiple features into fewer "principal components" that capture most of the variance
2. **Visualization** — reduce to 2 dimensions so we can plot clusters that exist in 3D (Age, Income, SpendingScore) on a 2D scatter plot

With only 3 features here PCA isn't strictly necessary for compression, but it's valuable for visualization and understanding what's driving the variation in the data.

### Results

Three principal components cover 100% of variance (expected since we only have 3 features):
- PC1: 44.27%
- PC2: 33.31%  
- PC3: 22.43%

PC1 and PC2 together explain 77.6% of variance — good enough to visualize clusters in 2D without losing too much information.

### What the Loadings Tell Us

The PCA loadings show how much each original feature contributes to each principal component:

- **PC1** is driven by Age (+0.706) and SpendingScore (-0.706) with almost no Income contribution. This means PC1 is essentially an "Age vs Spending" axis — older customers score high on PC1, high spenders score low.
- **PC2** is almost entirely Income (+0.999). It's practically just the income axis renamed.

This confirms what the correlation heatmap suggested — Income and SpendingScore are largely independent dimensions. Age and SpendingScore are the correlated pair (-0.33).

The 2D PCA plot shows the 5 clusters are reasonably well separated even in reduced dimensions — Clusters 1 and 2 (high spenders) are clearly on the left/negative PC1 side, Clusters 3 and 4 (careful and budget) are on the right/positive PC1 side.

---

## Step 4 — DBSCAN Clustering

### Why DBSCAN?

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) works completely differently from K-Means:

- K-Means: "I'll force every point into exactly K groups"
- DBSCAN: "I'll find dense regions and anything that doesn't fit in a dense region is noise"

DBSCAN doesn't need you to specify K upfront. It finds clusters based on density — points that are close together form a cluster, points in sparse regions become "noise" (outliers).

### Results — eps=0.3, min_samples=3

Found **7 clusters and 35 noise points (17.5% of customers).**

This is more fragmented than K-Means (5 clusters) but tells an interesting story:

**The 35 noise points are genuinely unusual customers** — they don't fit neatly into any dense group. These could be customers with unusual income/spending combinations that don't match the typical patterns. From a business perspective these might actually be the most interesting customers to investigate individually.

### K-Means vs DBSCAN — Which is Better?

Neither is universally better — they answer different questions:

**K-Means is better for this dataset** because the clusters are roughly spherical and similarly sized. K-Means gives cleaner, more actionable segments for marketing purposes — every customer gets assigned to exactly one group.

**DBSCAN adds value by identifying outliers** — the 35 noise customers that K-Means forces into a group anyway. These outlier customers might warrant individual attention or a separate "unknown" marketing category.

The honest conclusion: this dataset was designed for K-Means style clustering. DBSCAN struggles because the clusters aren't separated by density gaps — they're separated by direction (income vs spending axis) which is a K-Means strength, not DBSCAN's.

---

## Step 5 — SVM Classification

### The Idea

Once K-Means gave us 5 customer segments, I treated those labels as a supervised learning problem — can an SVM learn to classify new customers into the correct segment based on their Age, Income and SpendingScore?

This is a practical use case: once you've run clustering on existing customers, you want to automatically assign new customers to a segment without running the full clustering again.

### Why SVM?

SVM (Support Vector Machine) finds the optimal decision boundary between classes by maximizing the margin — the gap between the boundary and the nearest points of each class. It's particularly good at:
- Multi-class problems (we have 5 segments)
- Cases where classes are clearly separable (which our segments are)
- High-dimensional data

### Results

**Default SVM (RBF kernel, C=1.0): 97.50% accuracy**

The confusion matrix showed only 1 misclassification out of 40 test customers — one Cluster 3 customer was incorrectly assigned to Cluster 0. Given how well-separated the 5 segments are visually, this high accuracy makes sense.

Kernel comparison:
- Linear: 96.00% (±4.64%)
- RBF: 96.00% (±2.00%)  
- Polynomial: 90.00% (±4.47%)
- Sigmoid: 94.50% (±6.00%)

RBF kernel had the lowest standard deviation (2.00%) — most consistent performance. Linear was equally accurate but more variable. Polynomial performed worst, likely because the decision boundaries between segments don't need high-degree polynomial curves.

---

## Step 6 — RandomizedSearchCV Tuning

### RandomSearch vs GridSearch

I used RandomizedSearchCV here instead of GridSearchCV (which I used in Titanic). The difference:

- **GridSearch**: tries every single combination in the parameter grid — exhaustive but slow
- **RandomSearch**: randomly samples N combinations from the parameter distributions — faster, often finds equally good results

With continuous parameters like C and gamma (which can be any value, not just a few options), RandomSearch is actually better than GridSearch — there's no point testing exactly C=1, C=2, C=3 when the optimal might be C=1.47.

I defined distributions to sample from:
- C: uniform between 0.1 and 100
- gamma: uniform between 0.001 and 1
- kernel: ['rbf', 'linear', 'poly']

Tested 50 random combinations across 5 folds = 250 fits total.

### Interesting Finding — Tuning Made It Worse

**Best CV Score from RandomSearch: 99.38%**
**Tuned SVM test accuracy: 95.00%**
**Default SVM test accuracy: 97.50%**

The tuned model performed WORSE on the actual test set despite scoring higher in cross-validation. This is a real phenomenon called **overfitting to the validation set** — the RandomSearch found parameters that happened to work very well on those specific CV folds but didn't generalize as well to truly unseen data.

This happens especially with small datasets (only 200 customers) where there's high variance between different splits. The default C=1.0 with RBF kernel was already near-optimal for this data — there wasn't meaningful room to improve.

The right decision here is to keep the default SVM. This is an honest and important finding — sometimes tuning doesn't help, and recognizing that is just as valuable as when tuning does help.

---

## Final Results Summary

| Method | Result | Notes |
|--------|--------|-------|
| K-Means (K=5) | Silhouette: 0.5547 | Clean 5-segment solution |
| PCA (2 components) | 77.6% variance explained | Clusters visible in 2D |
| DBSCAN (eps=0.3) | 7 clusters, 35 outliers | Better for outlier detection |
| SVM (default) | 97.50% accuracy | Near-perfect classification |
| RandomizedSearchCV | 95.00% (worse) | Default was already optimal |

### The 5 Customer Segments

| Segment | Size | Strategy |
|---------|------|----------|
| Regular Customers | 81 | Maintain loyalty programs |
| Target Customers | 39 | Premium campaigns, VIP treatment |
| Impulsive Buyers | 22 | Flash sales, trend-based marketing |
| Careful Spenders | 35 | Build trust, quality messaging |
| Budget Customers | 23 | Value deals, discounts |

---

## Key Lessons

**1. Unsupervised learning requires more interpretation than supervised.** There's no accuracy score to optimize — you have to look at the clusters and decide if they make business sense. The 5 customer segments only became meaningful when I labelled them based on the cluster profiles.

**2. Different algorithms answer different questions.** K-Means said "here are 5 clean segments." DBSCAN said "here are some groups AND 35 customers who don't fit anywhere." Both are correct answers to different questions. Knowing which question you're asking determines which algorithm to use.

**3. PCA is about understanding, not just compression.** The loadings told me that Income and SpendingScore are independent dimensions — which explained why the clusters form the pattern they do. That's insight, not just dimensionality reduction.

**4. Tuning doesn't always help.** The RandomizedSearchCV result that performed worse than default is an important honest finding. With a small, clean, well-separated dataset the default parameters were already near-optimal. Recognizing when NOT to over-tune is a real skill.

**5. Visual EDA before modelling saves time.** The Income vs SpendingScore scatter plot showed 5 groups before any algorithm ran. That visual intuition guided the entire analysis — I already knew K=5 was likely correct before running the elbow method.

