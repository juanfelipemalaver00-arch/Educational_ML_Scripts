# Predictive Analytics & Machine Learning Curriculum in R

A comprehensive, production-grade repository of statistical machine learning algorithms, resampling strategies, market basket analytics, and automated time-series forecasting designed for predictive analytics and business intelligence.

---

##  Repository Architecture

```
Educational_ML_Scripts/
├── 01. Balancing and Confussion Matrix.R      # Imbalance Remediation & Resampling (ROSE, CNN, SMOTE)
├── 02. Log_Regression.R                      # Logistic Regression, Stepwise AIC & Profit Thresholding
├── 03. Trees_Clasification_RandomF.R         # CART, Pruning, CTree & Tuned Random Forest (Ranger)
├── 04. Product Recommendations -Upsell.R     # Association Rule Mining (Apriori) & Recommender Engine
├── 05. Time Series.R                         # Multi-Channel Forecasting Tournament (SARIMA vs Prophet) & Shiny App
│
├── APRIORI.xlsx                              # Transactional shopping basket matrix
├── ASOMEDIO2012A2025MES.csv                  # Monthly media advertising investments (2012–2025)
├── carteraguia2017.xlsx                      # Healthcare B2B default risk & accounts receivable portfolio
├── cerealdesayuno880.xlsx                    # Full cereal consumer dataset
├── cerealtest.csv                            # Cereal consumer choice test partition (Holdout)
├── cerealtrain.csv                           # Cereal consumer choice training partition
├── cosmeticspan.xlsx                         # Cosmetic retail Point-of-Sale (POS) transactions
└── financierosatisfaccion.xlsx               # Retail banking customer satisfaction dataset
```

---

## 🚀 Key Engineering & Analytical Standards

1. **Zero-Friction Package Management:**  
   Every script utilizes `pacman::p_load(...)` to dynamically verify, compile, and attach all dependencies without manual `install.packages()` overhead.
2. **Centralized Configuration (`CONFIG` Blocks):**  
   File paths, reproducibility seeds (`set.seed`), model hyperparameters ($k$, $CP$, $m_{try}$, $B$, $S$), and financial cost matrices are consolidated at the top of each script.
3. **Rigorous Data Science Foundations:**  
   In-depth pedagogical comments provide theoretical formulations for loss functions, distance metrics ($L_2$ norm), information gain / Gini impurity, logit link functions, and Fourier series decompositions.
4. **Actionable Commercial Impact:**  
   Direct translation of technical evaluation metrics (ROC-AUC, Macro-F1, Precision-Recall, Lift, MAPE) into business ROI, portfolio risk mitigation, and capital allocation.

---

## 🔬 Detailed Module Overview

### [01. Balancing and Confussion Matrix.R](file:///01.%20Balancing%20and%20Confussion%20Matrix.R)
* **Business Domain:** Healthcare B2B Credit Risk & Accounts Receivable Management (`carteraguia2017.xlsx`).
* **Problem:** Severe class imbalance (~15% default prevalence) causes naive classifiers to default to the majority class, missing high-risk debt accounts.
* **Techniques Implemented:**
  - **Random Under-Sampling & Over-Sampling (`ROSE`):** Baseline heuristic resampling.
  - **Condensed Nearest Neighbor (CNN):** Hart’s 1-NN algorithm to retain minimal boundary prototypes while pruning redundant interior majority instances.
  - **Synthetic Minority Over-sampling Technique (SMOTE):** Convex linear interpolation in feature space ($x_{new} = x_i + \lambda(x_{zi} - x_i)$) via `smotefamily`.
* **Output:** Comparative distribution audit and visual resampling benchmarks.

---

### [02. Log_Regression.R](file:///02.%20Log_Regression.R)
* **Business Domains:**
  1. Retail Banking Customer Advocacy (`financierosatisfaccion.xlsx`).
  2. B2B Delinquency Prediction & Credit Scoring (`carteraguia2017.xlsx`).
* **Techniques Implemented:**
  - **Binomial Generalized Linear Model (GLM):** Maximum Likelihood Estimation with logit link $\ln(p / (1-p)) = \mathbf{x}^T \boldsymbol{\beta}$.
  - **Stepwise Feature Selection:** Minimizing Akaike Information Criterion ($\text{AIC} = 2k - 2\ln L$).
  - **Vectorized Odds Sensitivity:** Matrix-broadcasted cumulative odds and probability shifts across parameter standard deviations.
  - **Financial Cost-Benefit Profit Optimization:**
    $$\text{Net Profit}(c) = P(Y=1) \cdot \text{Recall}(c) \cdot \left( \text{Benefit}_{\text{Recovered}} - \frac{\text{Cost}_{\text{Intervention}}}{\text{Precision}(c)} \right)$$
    Replaces arbitrary $0.50$ thresholds with the exact cut-off that maximizes recovered cash.

---

### [03. Trees_Clasification_RandomF.R](file:///03.%20Trees_Clasification_RandomF.R)
* **Business Domain:** Fast-Moving Consumer Goods (FMCG) / Food & Beverage Consumer Segmentation (`cerealtrain.csv`, `cerealtest.csv`).
* **Techniques Implemented:**
  - **CART Decision Trees (`rpart`):** Recursive binary splitting maximizing Gini Impurity reduction ($\Delta I_G$).
  - **Cost-Complexity Pruning ($CP$):** Selecting the optimal tree via the 1-SE rule on 10-fold cross-validation error ($xerror$).
  - **Conditional Inference Trees (`party::ctree`):** Permutation-based hypothesis testing ($p < 0.05$) to eliminate categorical variable selection bias.
  - **Tuned Random Forest (`ranger` + `caret`):** Multi-threaded grid search over `mtry` and `min.node.size` to minimize ensemble variance:
    $$\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2 \xrightarrow{B \to \infty, \rho \to 0} 0$$

---

### [04. Product Recommendations -Upsell.R](file:///04.%20Product%20Recommendations%20-Upsell.R)
* **Business Domain:** Cosmetics Retail & E-Commerce Basket Analytics (`cosmeticspan.xlsx`).
* **Techniques Implemented:**
  - **Sparse Matrix Encoding:** Converting transactional tables into Compressed Sparse Column format (`ngCMatrix`) via `arules`.
  - **Contingency Testing & Mosaic Visuals:** Standardized Pearson residuals ($r_{ij} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij}}}$) using `gmodels` and `vcd`.
  - **Apriori Association Rule Mining:** Downward-closure anti-monotonicity pruning for Support, Confidence, Lift, and Conviction.
  - **Network & Matrix Visualizations:** Static and interactive graph layouts using `arulesViz`.
  - **Production CSV Export:** Automated export of high-affinity rules for checkout microservices.

---

### [05. Time Series.R](file:///05.%20Time%20Series.R)
* **Business Domain:** Macro Media Advertising Expenditures across 7 Channels (`ASOMEDIO2012A2025MES.csv`, 2012–2025).
* **Techniques Implemented:**
  - **Multi-Series Descriptive Profiling:** Skewness, kurtosis, and decoupled trend visualizations across volatile channels.
  - **Automated Forecasting Tournament Engine:**
    - **Model A:** Box-Jenkins Seasonal Auto-ARIMA with exact Gaussian log-likelihood (`forecast::auto.arima`).
    - **Model B:** Meta Prophet Additive Model with Fourier seasonality (`prophet::prophet`).
    - **Selection:** Out-of-sample holdout tournament comparing test set MAPE and RMSE.
  - **Forward Projection:** 6-month predictions with 80% and 95% confidence ribbons exported to a consolidated Excel workbook (`writexl`).
  - **Interactive Shiny Dashboard:** Dynamic UI/Server application for Year-over-Year (YoY) budget variance auditing.

---

## 🛠️ Prerequisites & Setup

### Environment Requirements
- **R:** Version 4.0.0 or higher.
- **RStudio:** Recommended for interactive execution and Shiny hosting.

### Quick Start
1. Clone or download the repository to your local workspace:
   ```bash
   git clone https://github.com/juanfelipemalaver00-arch/Educational_ML_Scripts.git
   ```
2. Open RStudio and set your working directory to the project folder:
   ```R
   setwd("path/to/Educational_ML_Scripts")
   ```
3. Open and run any script directly. `pacman` will automatically handle package installation:
   ```R
   source("01. Balancing and Confussion Matrix.R")
   ```

---

## 📊 Summary of Analytical Frameworks

| Analytical Challenge | Classical Limitation | Implemented Modern Solution | Primary Metric |
| :--- | :--- | :--- | :--- |
| **Class Imbalance** | Naive accuracy trap; high False Negatives | SMOTE Convex Interpolation & Hart's CNN | Recall, Specificity, AUC |
| **Credit Default Scoring** | Arbitrary 0.50 cutoff; high cost of defaults | Cost-Benefit Expected Net Profit Curve | Net Profit ($/account) |
| **Consumer Choice Trees** | Overfitting and multi-level categorical bias | 1-SE $CP$ Pruning & Permutation CTree | Macro-F1, Holdout Accuracy |
| **Cross-Selling & Bundles** | Exponential $O(2^D)$ itemset search space | Apriori Downward Closure & Sparse Matrices | Support, Confidence, Lift |
| **Advertising Forecasting** | Overfitting structural shifts / seasonality | Out-of-sample SARIMA vs. Prophet Tournament | Holdout MAPE (%), RMSE |
