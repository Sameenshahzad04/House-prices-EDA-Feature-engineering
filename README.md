# 🏡 Ames Housing Price Prediction & Feature Engineering Pipeline

An end-to-end data science and feature engineering pipeline on the Ames Housing Dataset. This project systematically transforms raw tabular housing data (1,460 rows $\times$ 81 features) into a high-signal 144-column numerical feature matrix optimized for tree-based ensemble models.

---

## 📁 Repository Structure

```text
.
├── 📄 data/
│   ├── train.csv                      # Raw Ames Housing dataset (1,460 rows × 81 cols)
│   ├── train_processed_phase5.csv     # Output post-feature engineering
│   ├── train_processed_phase6.csv     # Output post-categorical encoding (144 cols)
│   └── train_processed_phase8.csv     # Final processed dataset for modeling
├── 📓 notebooks/
│   ├── EDA_phase3.ipynb               # Target analysis, missingness & univariate EDA
│   ├── EDA_phase_4_Multivariant.ipynb # Bivariate EDA & multicollinearity reduction
│   ├── FE_phase_5.ipynb               # Footprint aggregation & age engineering
│   ├── ECF_phase_6.ipynb              # Ordinal mapping, bit-encoding & dummy creation
│   └── phase7_phase8.ipynb            # Log transforms, feature selection & RF baseline
└── 📄 README.md                          # Repository documentation

```

---

## ⚡ Pipeline Architecture & Phase Summary

### 🎯 Phase 0 & 1: Data Understanding & Target Normalization

* **Data Classification:** Categorized all 81 raw features into continuous, discrete, ordinal, and nominal types to prevent silent downstream errors (e.g., mistaking structural `NA` for missing data).


* **Target Transformation:** Identified severe right skewness in `SalePrice` (skewness = 1.881, kurtosis = 6.510). Applied $\text{log1p}$ transformation to yield `SalePrice_Log` (skewness = 0.121, kurtosis = 0.803), stabilizing residual loss functions.



### 🔍 Phase 2: Missing Data Diagnostics

* **Structural Imputation (`NA` = Feature Absent):** Filled categorical structural nulls with `"None"` (`PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `Garage*`, `Bsmt*`) and numerical structural nulls with `0` (`MasVnrArea`).


* **MCAR Imputation:** Applied median imputation to `LotFrontage` (17.74% missing) and modal imputation to `Electrical`.



| Feature | Missing % | Classification | Imputation Strategy |
| --- | --- | --- | --- |
| **`PoolQC`** | 99.52%

 | Structural | Constant `"None"`<br> |
| **`MiscFeature`** | 96.30%

 | Structural | Constant `"None"`<br> |
| **`Alley`** | 93.77%

 | Structural | Constant `"None"`<br> |
| **`Fence`** | 80.75%

 | Structural | Constant `"None"`<br> |
| **`LotFrontage`** | 17.74%

 | Random (MCAR) | Median Imputation

 |

### 🧹 Phase 3 & 4: Anomaly Trimming & Multicollinearity

* **Outlier Handling:** Removed 4 data-entry anomalies in `GrLivArea` (> 4,000 sq ft selling at abnormally low prices). Capped extreme tails in `GarageArea` (938.25 sq ft) and `TotalBsmtSF` (2,052 sq ft) at their $1.5 \times \text{IQR}$ upper bounds. Dataset updated to 1,456 rows $\times$ 79 columns.


* **Collinearity Resolution ($\vert{}r\vert{} > 0.80$):** Retained higher $y$-correlated feature per redundant pair:


* Kept **`GarageCars`** over `GarageArea` ($r = 0.893$)


* Kept **`GrLivArea`** over `TotRmsAbvGrd` ($r = 0.834$)


* Kept **`YearBuilt`** over `GarageYrBlt` ($r = 0.825$)





### 🧬 Phase 5 & 6: Domain Feature Engineering & Categorical Encoding

* **Total Effective Footprint:** Created aggregate feature `Total EffectiveSF` = `GrLivArea` + `TotalBsmtSF` + `WoodDeckSF` + `OpenPorchSF` + `EnclosedPorch` + `3SsnPorch` + `ScreenPorch`, increasing target correlation from 0.7205 to 0.8199.


* **Temporal Dynamics:** Derived `HouseAge` ($\text{YrSold} - \text{YearBuilt}$), `RemodAge` ($\text{YrSold} - \text{YearRemodAdd}$), and `IsRemodeled` flag before dropping raw calendar years.


* **Proximity Flags:** Parsed `Condition1` and `Condition2` into targeted binary flags (`Has_Pos_Amenity`, `Has_Traffic_Disturbance`, `Has_Railroad_Disturbance`) to preserve economic directionality.


* **Encoding Matrix:**
* **Explicit Ordinal Mapping:** Mapped 11 quality/condition features on integer scales (0–5).


* **Neighborhood Bit-Encoding:** Compressed 25 high-cardinality neighborhood categories into 5 bit columns (`Nbhd_bit0`–`Nbhd_bit4`).


* **One-Hot Encoding:** Applied dummy encoding (`drop_first=True`) to remaining low-cardinality nominals, yielding 144 columns.





### 🌲 Phase 7 & 8: Transformations & Baseline Modeling

* Applied $\text{log1p}$ transformations to skewed predictors (`LotArea_log`, `LotFrontage_log`). Bypassed scaling steps to maintain raw split boundaries for decision trees.



---

## 📈 Baseline Model Performance

Evaluated using a baseline `RandomForestRegressor` on the original dollar scale via `np.expm1()`:

| Metric | Score |
| --- | --- |
| **Training RMSE** | **$18,800.66**<br> |
| **Validation RMSE** | **$23,517.16**<br> |
| **Overfitting Gap** | **~$4,717.00**<br> |

### 🏆 Top Feature Importances

```text
Total EffectiveSF  ████████████████████████████████████ 45.32%[cite: 7]
Overall Qual       ███████████████████████████ 36.91%[cite: 7]
HouseAge           █ 2.41%[cite: 7]
GarageCars         █ 1.49%[cite: 7]
GrLivArea          █ 1.39%[cite: 7]

```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/house-prices-eda-fe.git
cd house-prices-eda-fe

# 2. Set up virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn

```

---

## 💡 Key Lessons Learned

* **Domain-First Imputation:** Filling missing values based on data dictionaries (structural vs. random missingness) prevents fake feature creation.


* **Tree-Focused Preprocessing:** Skipping global normalization and relying on rank-preserving ordinal transformations maintains model interpretability without sacrificing tree performance.
