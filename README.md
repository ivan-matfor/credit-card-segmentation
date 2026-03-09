# Credit Card Customer Segmentation

#### -- Project Status: [Completed]

## Project Intro/Objective
The purpose of this project is to segment credit card customers into interpretable behavioural groups using unsupervised learning. The analysis is based on anonymized customer behaviour data over a six-month period and is designed to identify meaningful usage patterns that could support CRM, targeting, product strategy, and risk monitoring. The workflow goes beyond basic clustering by combining exploratory analysis, feature engineering, model comparison, centroid-based interpretation, relative cluster heatmaps, and anomaly detection. The final result is presented as a portfolio-ready analytical product with a clear business narrative.

### Methods Used
* Exploratory Data Analysis (EDA)
* Missing Value Imputation
* Feature Engineering
* Log Transformation
* Standardization / Scaling
* K-Means Clustering
* Elbow Method
* Silhouette Analysis
* Gaussian Mixture Models
* Centroid Interpretation
* Relative Cluster Heatmaps
* Isolation Forest Anomaly Detection
* PCA Visualization

### Technologies
* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* scikit-learn

## Project Description
This project analyzes anonymized credit card customer behaviour using a public dataset originally sourced from Kaggle:

**Source dataset:** [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

The version used in this repository is a slightly modified and processed version of that dataset, stored locally as:

`data/processed/card_transactions_kaggle.csv`

The dataset contains variables related to:

* account balances and credit limits
* purchases and purchase frequencies
* installment behaviour
* cash advances
* payment and minimum payment behaviour

The notebook is structured around these **10 main steps**:

1. **Dataset**  
   The project documents the dataset origin, explains that it comes from Kaggle, and describes the business meaning of the available fields.

2. **Loading Dataset**  
   The processed dataset is loaded from `data/processed/`. Missing values in `credit_limit` and `min_payments` are handled with median imputation.

3. **EDA**  
   The project explores the structure of the data through summary statistics, feature inspection, a correlation heatmap, and bar charts. This reveals skewed variables, redundancy, and natural behavioural blocks.

4. **Feature Engineering**  
   New behaviour-driven variables are created to improve interpretability, including:
   * `avg_purchase`
   * `installment_ratio`
   * `cash_advance_ratio`
   * `balance_to_limit`
   * `payment_minpay_ratio`

   Redundant original columns are then dropped to simplify the feature space.

5. **K-Means Model**  
   K-Means is trained as the main segmentation model after preprocessing the feature space.

6. **Gaussian Mixture Models with 4 Clusters**  
   A GMM model is added as a comparison layer to check whether the main behavioural structure is broadly consistent under a different clustering assumption.

7. **Isolation Forest Anomaly Detection**  
   Isolation Forest is used to detect unusual customers and check how anomalies are distributed across the discovered segments.

8. **Interpreting Clusters Using Centroids**  
   Cluster centroids are transformed back to the original feature scale so the segment profiles can be interpreted in business terms.

9. **Summary Cluster Behavior Heatmap (K-Means)**  
   A relative heatmap compares each cluster against the overall dataset mean, making it easier to understand which features are above or below average in each segment.

10. **Visualizing Anomaly Detection Behavior (Isolation Forest)**  
    PCA is used to project the feature space into two dimensions and visually inspect how detected anomalies sit within the broader customer structure.

The final K-Means segmentation is interpreted as:

* **Active Revolvers**
* **Light Users**
* **High Value Transactors**
* **Cash Advance Users**

## Getting Started

1. Clone this repository.

2. In this repository:
   * processed files are stored in `data/processed/`

3. The main notebook is located at:
   * `notebooks/Customer segmentation challenge.ipynb`

4. Suggested environment setup:

```bash
conda create -n customer-segmentation python=3.10 -y
conda activate customer-segmentation
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter lab
```

## Featured Notebooks/Analysis/Deliverables
* [Main notebook - Customer segmentation challenge](notebooks/Customer%20segmentation%20challenge.ipynb)
* [Project report](reports/report.md)
* [Processed dataset](data/processed/card_transactions_kaggle.csv)

## Key Results

* The analysis supports a **clear 4-cluster customer structure**.
* Feature engineering improved interpretability by shifting from raw operational variables to behaviour-oriented ratios.
* K-Means and Gaussian Mixture Models produced broadly consistent behavioural patterns.
* Isolation Forest was applied with a **2% contamination parameter**, identifying **179 anomalous customers**, with anomalies concentrated primarily in the **Active Revolvers** and **Cash Advance Users** segments.
* The project combines **EDA, feature engineering, unsupervised modelling, business interpretation, and anomaly detection** in one coherent workflow.

## Future Improvements

* Save final cluster assignments as an exportable CSV for downstream business use
* Add a formal preprocessing script or pipeline under `src/` to make the workflow reusable outside the notebook
* Test DBSCAN or hierarchical clustering for alternative cluster shapes

## Contributing Members

[Ivan Mateo Forcen](https://github.com/ivan-matfor)

#### Other Members:

