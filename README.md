# 🏡 Ames Housing: Reasoning-First EDA & Domain Feature Engineering

An end-to-end Machine Learning pipeline built on the **Ames Housing Dataset**. This project focuses on **deep domain reasoning** behind data transformations—explaining the explicit *why* for every decision, from target distribution scaling to model-specific preprocessing (Linear Regression vs. XGBoost / Random Forest).

---

## 📌 Project Overview & Deliverables

Rather than blindly throwing automated transformations or one-hot encoding at the dataset, this project systematically breaks down all **80 features** into a domain-driven pipeline:

1. **Target Analysis:** Analyzing $Y$ skewness/kurtosis and stabilizing variance via log-transformation ($\log(1+x)$).
2. **Structural Missingness vs. Unobserved Data:** Treating missing values based on architectural/domain rules (e.g., `PoolQC = NaN` means *No Pool*, not missing data).
3. **Domain Feature Engineering:** Consolidating sparse categoricals and constructing composite physical metrics (Total SF, Quality-Area interactions, House Age).
4. **Model-Aware Pipeline Design:** Understanding why monotonic transformations and feature scaling are critical for Linear models but invariant for Tree-based models.

---

## 💡 What Makes This Solution Unique?

### 1. Dual Preprocessing Paths (Linear vs. Tree Models)
* **For Linear Models (Ridge/Lasso):** Skewed features are log-transformed ($\log(1+x)$) to fix heteroscedasticity, high-cardinality features are One-Hot Encoded, and values are scaled with `RobustScaler` to limit outlier impact.
* **For Tree Models (XGBoost & Random Forest):** Input feature skewness and scale transformations are bypassed because tree splits rely strictly on feature ranking ($X_j \le t$). However, **Target ($Y$) Transformation** is retained to prevent extreme luxury mansions from corrupting mean-squared error (MSE) leaf calculations.

### 2. Domain Feature Consolidation over Blind One-Hot Encoding
To prevent high-cardinality overfitting and sparse feature matrices, raw categories were engineered into dense, high-signal flags:
* **Overlapping Transaction Context:** Merged `SaleType` + `SaleCondition` into explicit financial flags: `Is_New_Construction`, `Is_Distressed_Sale`, and `Is_Normal_Sale`.
* **Structural Layout Extraction:** Deconstructed `HouseStyle` into a continuous `HouseStyle_Stories` metric and a binary `Is_Split_Layout` flag.
* **Heavy Dominance Filtering:** Replaced highly skewed categoricals (>98% single class like `RoofMatl` and `Heating`) with single binary indicator flags.

### 3. Derived Interaction Features
* **Total Living Space:** $\text{TotalSF} = \text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$
* **Composite Bathrooms:** $\text{TotalBath} = \text{FullBath} + 0.5(\text{HalfBath}) + \text{BsmtFullBath} + 0.5(\text{BsmtHalfBath})$
* **Domain Interaction (Invented):** $\text{OverallQual\_x\_TotalSF} = \text{OverallQual} \times \text{TotalSF}$ (captures price scaling per square foot relative to material finish quality).

---

## 📊 Pipeline Phases

The project is structured in a single Jupyter Notebook across **9 distinct phases**:

| Phase | Description | Key Outputs |
| :--- | :--- | :--- |
| **Phase 0** | Setup & Manual Column Classification | Classified all 80 features into Continuous, Discrete, Ordinal, Nominal, and Target. |
| **Phase 1** | Target Variable Analysis | Analyzed `SalePrice` skewness (1.88) & kurtosis (6.53); applied $\log(1+p)$ transform. |
| **Phase 2** | Missing Data Decision Matrix | Differentiated structural missingness (`None`/`0`) from random nulls (Median/Mode). |
| **Phase 3** | Univariate EDA & Outliers | Flagged continuous skewness, identified rare categorical levels (<1%), evaluated IQR outliers. |
| **Phase 4** | Multivariate EDA & Collinearities | Identfied top price drivers (`OverallQual`, `GrLivArea`, `TotalBsmtSF`) and collinear pairs. |
| **Phase 5** | Derived Feature Engineering | Engineered `TotalSF`, `HouseAge`, `TotalBathrooms`, and `OverallQual_x_TotalSF`. |
| **Phase 6** | Ordinal & Nominal Encoding | Mapped ordinal qualities (`Ex` $\rightarrow$ 5, `Po` $\rightarrow$ 1); bucketed low-frequency nominals into "Other". |
| **Phase 7** | Scaling & Model Differentiation | Compared `StandardScaler` vs. `RobustScaler`; documented Linear vs. Tree-model requirements. |
| **Phase 8** | Feature Selection & Validation | Removed redundant pairs; computed baseline `RandomForestRegressor` feature importances. |

---

## 📝 Key Findings & Executive Summary

### Top 3 EDA Findings
1. **Target Distribution is Right-Skewed:** `SalePrice` had a right skewness of **1.88** and excess kurtosis of **6.53**. Applying `log1p` pulled skewness down to **0.12**, yielding a near-normal distribution.
2. **Missingness is Structural, Not Random:** Over 90% of missing values in columns like `PoolQC`, `MiscFeature`, `Alley`, `Fence`, and `GarageType` represent the **absence of a feature** rather than missing data. Treating them as `None` or `0` preserved critical physical context.
3. **Severe Multicollinearity in Area/Garage Features:** `GarageCars` vs. `GarageArea` ($r = 0.88$) and `TotalBsmtSF` vs. `1stFlrSF` ($r = 0.81$) were highly collinear.

### Top 3 Engineered Features
1. **`TotalSF`:** Higher correlation with `SalePrice` ($r \approx 0.78$) than any single raw area feature.
2. **`OverallQual_x_TotalSF`:** Ranked among the top 3 features in Random Forest feature importance, effectively modeling price elasticity.
3. **`HouseAge` ($\text{YrSold} - \text{YearBuilt}$):** Provided a linear decay signal far stronger than raw construction year inputs.

---
🚀 How to Run Locally
1. Clone the repository
Bash
git clone [https://github.com/YOUR_USERNAME/house-prices-eda-feature-engineering.git](https://github.com/YOUR_USERNAME/house-prices-eda-feature-engineering.git)
cd house-prices-eda-feature-engineering
2. Set up virtual environment
Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install dependencies
Bash
pip install -r requirements.txt
