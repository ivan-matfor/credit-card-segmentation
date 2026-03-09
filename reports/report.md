# Credit Card Customer Segmentation Report

# Summary

**Motivation:** Credit card customer datasets usually contain several distinct behaviour patterns hidden behind raw operational variables. A segmentation workflow helps transform those variables into interpretable profiles that can be useful for business analysis, CRM, targeting, and risk monitoring.

**Goal:** Build an unsupervised learning workflow that segments anonymized credit card customers into meaningful behavioural groups and identifies unusual customers through anomaly detection.

**What we propose:** We use a processed version of a public Kaggle credit card dataset, perform exploratory analysis, engineer behaviour-based ratios, apply log transformation and scaling, build a K-Means clustering model, compare it with a Gaussian Mixture Model, interpret the segments with centroids and relative heatmaps, and finally detect anomalies with Isolation Forest supported by PCA visualization. The result is a 4-cluster segmentation with interpretable profiles: Active Revolvers, Light Users, High Value Transactors, and Cash Advance Users.

**Next Steps:** Export final cluster labels for downstream use, formalize preprocessing in reusable code under `src/`, test additional clustering methods, and build a simple dashboard for non-technical audiences.

**Structure of the document:** This report covers the dataset origin, exploratory analysis, feature engineering, clustering workflow, segment interpretation, anomaly detection, and final recommendations.

# Introduction

This project studies anonymized credit card customer behaviour over a six-month period using a public dataset originally sourced from Kaggle: [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).

The dataset includes variables related to balances, purchases, purchase frequencies, installment behaviour, cash advances, payments, and credit usage. The project does not try to predict a target variable. Instead, it aims to discover natural behavioural groups in the data.

This kind of problem is a strong use case for unsupervised learning. From a business point of view, customer segmentation can help identify different usage profiles, guide marketing actions, support product decisions, and highlight potentially risky or unusual behaviours. The analysis was therefore designed not only to produce clusters, but also to make them interpretable and portfolio-ready.

# Part 1 - Dataset, Loading, and Exploratory Analysis

The notebook begins by documenting the dataset source and the meaning of the available fields. The version used inside the repository is a processed local file stored at:

`data/processed/card_transactions_kaggle.csv`

After loading the data, the workflow performs initial inspection and basic cleaning. Missing values in `credit_limit` and `min_payments` are imputed with the median.

The exploratory stage then examines:

* dataset shape and data types
* summary statistics
* correlation structure
* feature distributions

Two findings are especially important:

* many monetary and count variables are strongly skewed
* several features are highly correlated and therefore partially redundant

The heatmap also suggests that the data naturally forms behavioural blocks related to spending, cash advance usage, repayment, and credit utilization. This makes a strong case for feature engineering before clustering.

# Part 2 - Feature Engineering and Preprocessing

A key part of the project is transforming raw activity measures into more interpretable behavioural ratios. The notebook creates the following derived features:

* `avg_purchase`
* `installment_ratio`
* `cash_advance_ratio`
* `balance_to_limit`
* `payment_minpay_ratio`

These variables make the segmentation more meaningful. For example, instead of only looking at purchase totals, the analysis can distinguish whether customers spend in installments, rely on cash advances, or carry balances relative to their credit limits.

Some original columns are dropped after feature engineering to reduce redundancy and keep the modelling space easier to interpret.

Because the project relies on distance-based clustering methods, preprocessing is essential. The notebook therefore applies:

* **log transformation** to strongly skewed variables
* **standardization** with `StandardScaler` so all modelling features contribute on a comparable scale

Without this step, high-magnitude variables would dominate the clustering process.

# Part 3 - Clustering Workflow

The core segmentation model is **K-Means**.

To decide how many clusters to keep, the notebook uses two common evaluation approaches:

* **Elbow method** to inspect inertia reduction
* **Silhouette analysis** to assess compactness and separation

The selected solution uses **4 clusters**, which offers a good trade-off between structure and interpretability.

The notebook then analyzes K-Means cluster profiles using grouped comparisons across features. This is followed by a complementary **Gaussian Mixture Model with 4 clusters**. The GMM solution is not treated as the final model, but as a robustness check. It shows broadly similar behavioural patterns, which increases confidence that the segmentation is not just an artefact of one algorithm.

# Part 4 – Cluster Interpretation Through Visual Analysis

After selecting the final K-Means segmentation, the first interpretation layer is built through visual analysis.

The notebook generates multiple cluster-level plots showing how key behavioural variables differ across the four segments. These visual comparisons help identify the dominant behaviour patterns inside each group before introducing more formal summaries.

By plotting variables such as purchase activity, installment usage, balance behaviour, and cash advance patterns across clusters, the analysis reveals clear behavioural differences between segments.

This visual inspection allows the clusters to be initially interpreted as:

## 1. Active Revolvers
Customers who actively use their cards and maintain stronger revolving credit behaviour. They tend to carry balances and show higher interaction with credit limits.

## 2. Light Users
Customers with consistently low activity across most variables. They use their cards occasionally and show minimal engagement with credit features.

## 3. High Value Transactors
Customers with higher purchase activity and stronger repayment behaviour, suggesting more controlled and higher-value transaction usage.

## 4. Cash Advance Users
Customers whose behaviour is dominated by frequent reliance on cash advances, indicating a very distinct usage pattern compared to the other groups.

This first interpretation layer relies purely on **visual cluster comparisons** and provides an intuitive understanding of the segmentation structure.

# Part 5 – Cluster Interpretation Using Centroids

To strengthen interpretability, the notebook converts the K-Means centroids from the scaled feature space back to the original feature scale.

This produces centroid tables that summarize the average behavioural profile of each segment.

Centroid analysis confirms and refines the patterns observed in the visual inspection stage, highlighting which behavioural indicators are above or below the dataset average for each segment.

This second interpretation layer transforms the segmentation into a clearer analytical summary and provides a structured comparison between clusters.

# Part 6 - Anomaly Detection and Visualization

The project goes one step further by adding **Isolation Forest**.

This is useful because clustering describes common structure, but does not directly isolate unusual observations. With a contamination rate of 2%, the notebook detects **179 anomalous customers**.

Those anomalies are then cross-checked against the discovered K-Means segments. The concentration is highest in:

* **Active Revolvers**
* **Cash Advance Users**

This suggests that those groups contain more behavioural heterogeneity or more extreme customer patterns than the other segments.

Finally, the notebook uses **PCA** to project the feature space into two dimensions and visualize where anomalous customers lie. This gives an intuitive view of how outliers sit at the edges or lower-density regions of the broader customer structure.

# Part 7 - Notebook Structure

The notebook is organized into these 10 main steps:

1. Dataset  
2. Loading Dataset  
3. EDA  
4. Feature Engineering  
5. K-Means Model  
6. Gaussian Mixture Models with 4 Clusters  
7. Isolation Forest Anomaly Detection  
8. Interpreting Clusters Using Centroids  
9. Summary Cluster Behavior Heatmap (K-Means)  
10. Visualizing Anomaly Detection Behavior (Isolation Forest)

This structure gives the project a clear analytical narrative from raw data understanding to interpretable unsupervised outputs.

# Conclusion & Recommendations

This project successfully turns an anonymized credit card dataset into an interpretable customer segmentation workflow. It combines exploratory analysis, feature engineering, preprocessing, unsupervised learning, model comparison, cluster interpretation, and anomaly detection in a way that is both technically coherent and easy to explain.

The strongest result is a robust **4-cluster segmentation** supported by multiple interpretation views and complemented by anomaly detection. From a portfolio perspective, the project is stronger than a basic clustering notebook because it shows structured thinking, model comparison, and business translation.

Recommended next actions:

* export final cluster assignments for downstream use
* create reusable preprocessing code in `src/`
* test DBSCAN or hierarchical clustering as alternative approaches
* add a small dashboard for stakeholder-facing communication
* package the workflow for reuse on similar customer datasets

# References

- Kaggle dataset source: [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
- scikit-learn documentation for K-Means, Gaussian Mixture Models, PCA, and Isolation Forest
- Project notebook: `notebooks/Customer segmentation challenge.ipynb`
