# ------------------------------------------------------------------------------
# 1. DATASET DESCRIPTION & BUSINESS DOMAIN FOUNDATIONS
# ------------------------------------------------------------------------------
# Dataset File: "carteraguia2017.xlsx"
# Domain: B2B Accounts Receivable & Default Risk in Healthcare (IPS Colombia).
#
# Business Reality & Financial Stake:
#   A medical supplier provides surgical materials and pharmaceuticals on credit 
#   to private and public healthcare providers (Instituciones Prestadoras de Salud - IPS).
#   Due to systemic cash-flow delays in the Colombian health system (ADRES/EPS transfers), 
#   client payment delays (`retrasos` = 1) threaten the supplier's working capital.
#
# Dataset Features:
#   - `PORC_PASIVOS_VENTA_ANUAL`: Total Liabilities / Annual Gross Sales (Debt Leverage Ratio).
#     Mathematical Intuition: High values indicate extreme debt service burden.
#   - `IGUAL_DUENO`: Binary/Continuous metric of corporate ownership stability.
#   - `TIPO_ips` / `TIPOips`: Facility tier (1: General, 2: Specialized, 3: High-complexity hospital).
#   - `cartera_actual_insumos_A/B`: Outstanding credit balance in Product Line A / B.
#   - `retrasos` (Target Variable): 1 = Significant payment delay / default; 0 = On-time payment.
#
# The Core Data Science Dilemma (Class Imbalance):
#   Defaults represent only ~15-20% of accounts. Standard Machine Learning loss functions
#   (Cross-Entropy / 0-1 Loss) penalize all misclassifications equally. Consequently, 
#   a trivial model predicting "0" for 100% of accounts attains 80-85% accuracy, 
#   yet generates massive financial losses by letting 100% of defaulting accounts slip through.
# ==============================================================================

# ------------------------------------------------------------------------------
# 2. ENVIRONMENT SETUP & DEPENDENCY MANAGEMENT
# ------------------------------------------------------------------------------
# Why 'pacman::p_load'?
# Standard 'library()' halts execution if a package is uninstalled.
# 'pacman::p_load()' performs an automated conditional check: if installed, it attaches;
# if missing, it compiles and installs from CRAN before attaching, ensuring zero-friction deployment.
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}

pacman::p_load(
  readxl,       # Optimized C++ binary parsing of Excel sheets (.xlsx)
  dplyr,        # Tidy grammar for relational transformations
  ggplot2,      # Coherent layered visual grammar (Wilkinson, 2005)
  scales,       # Scientific, percentage, and currency axis transformations
  ROSE,         # Random Over-Sampling Examples & Smoothed Bootstrap resampling
  class,        # Classical Nearest Neighbor classification algorithms
  smotefamily,  # Synthetic Minority Over-sampling Technique (Chawla et al., 2002)
  caret         # Classification And REgression Training (evaluation matrices)
)

# ------------------------------------------------------------------------------
# 3. GLOBAL CONFIGURATION & HYPERPARAMETERS
# ------------------------------------------------------------------------------
# Centralizing all parameters at the top avoids magic numbers, supports automated
# hyperparameter tuning, and guarantees exact scientific reproducibility.
CONFIG <- list(
  # File paths
  DATA_PATH       = "carteraguia2017.xlsx",
  
  # Reproducibility Seeds
  # Why fixed seeds? Pseudo-random generators (Mersenne Twister) require deterministic
  # starting states so colleagues, auditors, and regulators obtain identical sample splits.
  SEED_GLOBAL     = 126,
  SEED_SMOTE      = 42,
  
  # SMOTE Hyperparameters:
  # 'SMOTE_K' = 5: Number of Euclidean nearest neighbors considered for linear interpolation.
  #   - Low K (e.g., 1): Generates noise along outlier directions.
  #   - High K (e.g., 15): Risks interpolating across overlapping class boundaries.
  SMOTE_K         = 5,
  # 'SMOTE_DUP_SIZE' = 2: Minority duplication multiplier. 
  #   - 2 creates 200% additional synthetic minority cases, balancing the class ratio.
  SMOTE_DUP_SIZE  = 2,
  
  # Visual formatting
  THEME_BASE_SIZE = 12
)

set.seed(CONFIG$SEED_GLOBAL)

# ------------------------------------------------------------------------------
# 4. DATA INGESTION & IMBALANCE AUDIT
# ------------------------------------------------------------------------------
# Defensive programming: assert file existence before attempting I/O
if (!file.exists(CONFIG$DATA_PATH)) {
  stop(sprintf("Fatal Error: Dataset '%s' not found in working directory.", CONFIG$DATA_PATH))
}

raw_cartera <- read_excel(CONFIG$DATA_PATH)

# Data Preprocessing & Factor Encoding
# Why factor conversion? R's GLM and tree engines require categorical variables to be 
# explicitly typed as 'factor' to construct internal dummy contrast matrices.
cartera <- raw_cartera %>%
  mutate(
    retrasos = as.factor(retrasos),
    across(where(is.character), as.factor)
  )

# Inspect Baseline Imbalance Ratio: P(Y=1) vs P(Y=0)
cat("\n=== BASELINE CLASS DISTRIBUTION (PRIOR PROBABILITIES) ===\n")
baseline_counts <- table(cartera$retrasos)
baseline_props  <- prop.table(baseline_counts)
print(baseline_counts)
print(round(baseline_props * 100, 2))

# Visualizing the Imbalanced Feature Space
# Business Insight: Notice how the red defaulting accounts are sparse and surrounded 
# by non-defaulting accounts, making empirical decision boundaries biased toward blue.
p_baseline <- ggplot(cartera, aes(x = PORC_PASIVOS_VENTA_ANUAL, y = IGUAL_DUENO, color = retrasos)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_manual(
    values = c("0" = "#2b5c8f", "1" = "#d95f02"),
    labels = c("0" = "On-time Account (0)", "1" = "Delinquent Account (1)")
  ) +
  labs(
    title = "Baseline Imbalanced Credit Portfolio",
    subtitle = sprintf("Default Prevalence: %.2f%% | Severe boundary skew", baseline_props["1"] * 100),
    x = "Liabilities / Annual Sales Ratio (Debt Pressure)",
    y = "Ownership Continuity Index",
    color = "Payment Status"
  ) +
  theme_minimal(base_size = CONFIG$THEME_BASE_SIZE) +
  theme(legend.position = "top")

print(p_baseline)

# Extract continuous features for distance-based resampling algorithms
cart_numeric <- cartera %>%
  select(where(is.numeric), retrasos) %>%
  na.omit()

# ------------------------------------------------------------------------------
# 5. RESAMPLING METHOD 1: RANDOM UNDER-SAMPLING (ROSE)
# ------------------------------------------------------------------------------
# Why do we do this?
# In Random Under-Sampling, we artificially downsample the majority class (0) 
# until its frequency matches the minority class (1).
#
# Function Argument Breakdown:
#   - 'retrasos ~ .': R formula specifying that 'retrasos' is the target to balance.
#   - 'method = "under"': Subsamples majority instances without replacement.
#
# Mathematical Foundation:
#   Forces prior probability balance: P_resampled(Y=0) = P_resampled(Y=1) = 0.5.
#   Shifts Bayes decision threshold from: c = P(Y=0)/P(Y=1) to: c = 1.0.
#
# Trade-Off Analysis:
#   - Pros: Drastically reduces training runtime; eliminates bias toward majority.
#   - Cons: Throws away potentially critical majority data, increasing model variance.

set.seed(CONFIG$SEED_GLOBAL)
df_under <- ovun.sample(retrasos ~ ., data = cart_numeric, method = "under")$data

cat("\n=== RANDOM UNDER-SAMPLING CLASS COUNTS ===\n")
print(table(df_under$retrasos))

p_under <- ggplot(df_under, aes(x = PORC_PASIVOS_VENTA_ANUAL, y = IGUAL_DUENO, color = retrasos)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_manual(values = c("0" = "#2b5c8f", "1" = "#d95f02")) +
  labs(
    title = "Resampled Space: Random Under-Sampling",
    subtitle = "Majority class randomly pruned to reach 50/50 balance",
    x = "Liabilities / Annual Sales Ratio",
    y = "Ownership Continuity Index",
    color = "Status"
  ) +
  theme_minimal(base_size = CONFIG$THEME_BASE_SIZE) +
  theme(legend.position = "top")

print(p_under)

# ------------------------------------------------------------------------------
# 6. RESAMPLING METHOD 2: RANDOM OVER-SAMPLING (ROSE)
# ------------------------------------------------------------------------------
# Why do we do this?
# Replicates minority instances (delinquencies) with replacement until equal to majority.
#
# Function Argument Breakdown:
#   - 'method = "over"': Samples minority instances with replacement until sample sizes match.
#
# Mathematical Foundation:
#   Weights minority observations proportionally in the empirical risk objective:
#   min_theta sum_{i=1}^N w_i * L(y_i, f(x_i; theta))
#
# Trade-Off Analysis:
#   - Pros: Retains all majority information; easy to implement.
#   - Cons: Exact point replication leads classifiers (like Decision Trees) to create
#     hyper-specific rule partitions around single points (severe overfitting).

set.seed(CONFIG$SEED_GLOBAL)
df_over <- ovun.sample(retrasos ~ ., data = cart_numeric, method = "over")$data

cat("\n=== RANDOM OVER-SAMPLING CLASS COUNTS ===\n")
print(table(df_over$retrasos))

p_over <- ggplot(df_over, aes(x = PORC_PASIVOS_VENTA_ANUAL, y = IGUAL_DUENO, color = retrasos)) +
  geom_point(alpha = 0.5, size = 2) +
  scale_color_manual(values = c("0" = "#2b5c8f", "1" = "#d95f02")) +
  labs(
    title = "Resampled Space: Random Over-Sampling",
    subtitle = "Minority cases duplicated with replacement (Overlapping points)",
    x = "Liabilities / Annual Sales Ratio",
    y = "Ownership Continuity Index",
    color = "Status"
  ) +
  theme_minimal(base_size = CONFIG$THEME_BASE_SIZE) +
  theme(legend.position = "top")

print(p_over)

# ------------------------------------------------------------------------------
# 7. RESAMPLING METHOD 3: CONDENSED NEAREST NEIGHBOR (CNN)
# ------------------------------------------------------------------------------
# Why do we do this? (Advanced Data-Cleaning Undersampling)
# Developed by P. Hart (1968), CNN identifies and keeps only the "prototype" samples 
# that define the geometric decision boundary, while pruning redundant interior majority points.
#
# Mathematical Principle:
#   Let T be the full training set. We construct a minimal subset S subseteq T such that
#   1-Nearest Neighbor classification using S correctly classifies all points in T.
#   - If a point x_i is correctly classified by 1-NN on current S -> It is redundant (discard).
#   - If a point x_i is misclassified by 1-NN on current S -> It is a boundary case (add to S).
#
# Function Argument Breakdown:
#   - 'drop = FALSE': Prevents R from coercing a 1-row slice into a dimensionless vector,
#     maintaining the required 2D matrix structure for 'class::knn()'.
#   - 'k = 1': 1-Nearest Neighbor metric based on Euclidean distance:
#     d(x, z) = sqrt(sum_{j=1}^D (x_j - z_j)^2).

cnn_undersample <- function(feature_matrix, label_vector) {
  stopifnot(nrow(feature_matrix) == length(label_vector))
  
  # Step 1: Initialize subset S with 1 example from each class
  classes <- unique(label_vector)
  init_indices <- sapply(classes, function(c) which(label_vector == c)[1])
  
  subset_data   <- feature_matrix[init_indices, , drop = FALSE]
  subset_labels <- label_vector[init_indices]
  
  remaining_idx <- setdiff(seq_len(nrow(feature_matrix)), init_indices)
  
  # Step 2: Iterate over all remaining points
  for (idx in remaining_idx) {
    query_point <- feature_matrix[idx, , drop = FALSE]
    true_label  <- label_vector[idx]
    
    # Classify point using 1-NN on current condensed set S
    pred_label <- knn(
      train = subset_data,
      test  = query_point,
      cl    = subset_labels,
      k     = 1
    )
    
    # If the condensed set fails to classify the point, it carries boundary information
    if (pred_label != true_label) {
      subset_data   <- rbind(subset_data, query_point)
      subset_labels <- c(subset_labels, true_label)
    }
  }
  
  res_df <- as.data.frame(subset_data)
  res_df$retrasos <- factor(subset_labels, levels = levels(label_vector))
  return(res_df)
}

# Execute CNN
X_mat <- as.matrix(cart_numeric %>% select(-retrasos))
y_vec <- cart_numeric$retrasos

df_cnn <- cnn_undersample(X_mat, y_vec)

cat("\n=== CONDENSED NEAREST NEIGHBOR (CNN) CLASS COUNTS ===\n")
print(table(df_cnn$retrasos))

p_cnn <- ggplot(df_cnn, aes(x = PORC_PASIVOS_VENTA_ANUAL, y = IGUAL_DUENO, color = retrasos)) +
  geom_point(alpha = 0.8, size = 2.5) +
  scale_color_manual(values = c("0" = "#2b5c8f", "1" = "#d95f02")) +
  labs(
    title = "Resampled Space: Condensed Nearest Neighbor (CNN)",
    subtitle = "Only boundary prototypes retained; homogeneous interior majority pruned",
    x = "Liabilities / Annual Sales Ratio",
    y = "Ownership Continuity Index",
    color = "Status"
  ) +
  theme_minimal(base_size = CONFIG$THEME_BASE_SIZE) +
  theme(legend.position = "top")

print(p_cnn)

# ------------------------------------------------------------------------------
# 8. RESAMPLING METHOD 4: SMOTE (SYNTHETIC MINORITY OVER-SAMPLING)
# ------------------------------------------------------------------------------
# Why do we do this? (State-of-the-Art Synthetic Interpolation)
# Chawla et al. (2002) proposed generating *new synthetic minority examples* along 
# the feature segments connecting minority neighbors, rather than simply duplicating them.
#
# Mathematical Foundation:
#   For every minority sample x_i in X_minority:
#     1. Compute Euclidean distance to all minority points: d(x_i, x_j) = ||x_i - x_j||_2.
#     2. Select its K-nearest minority neighbors N_K(x_i).
#     3. Select a random neighbor x_zi in N_K(x_i).
#     4. Synthesize new sample via convex linear interpolation:
#        x_new = x_i + lambda * (x_zi - x_i),  where lambda ~ Uniform(0, 1).
#
# Function Argument Breakdown:
#   - 'X': Matrix/Dataframe of continuous numeric features.
#   - 'target': Factor vector containing binary target classes.
#   - 'K = 5': 5 nearest neighbors used to define the convex interpolation neighborhood.
#   - 'dup_size = 2': Generates 2 new synthetic cases per existing minority point (200% oversampling).

set.seed(CONFIG$SEED_SMOTE)
smote_output <- SMOTE(
  X        = cart_numeric %>% select(-retrasos),
  target   = cart_numeric$retrasos,
  K        = CONFIG$SMOTE_K,
  dup_size = CONFIG$SMOTE_DUP_SIZE
)

# Extract and standardize SMOTE output
df_smote <- smote_output$data %>%
  rename(retrasos = class) %>%
  mutate(retrasos = as.factor(retrasos))

cat("\n=== SMOTE RESAMPLED CLASS DISTRIBUTION ===\n")
print(table(df_smote$retrasos))
print(round(prop.table(table(df_smote$retrasos)) * 100, 2))

p_smote <- ggplot(df_smote, aes(x = PORC_PASIVOS_VENTA_ANUAL, y = IGUAL_DUENO, color = retrasos)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_manual(values = c("0" = "#2b5c8f", "1" = "#d95f02")) +
  labs(
    title = "Resampled Space: SMOTE (Synthetic Convex Hull)",
    subtitle = sprintf("K=%d, dup_size=%d | Expands minority decision boundary", 
                       CONFIG$SMOTE_K, CONFIG$SMOTE_DUP_SIZE),
    x = "Liabilities / Annual Sales Ratio",
    y = "Ownership Continuity Index",
    color = "Status"
  ) +
  theme_minimal(base_size = CONFIG$THEME_BASE_SIZE) +
  theme(legend.position = "top")

print(p_smote)

# ------------------------------------------------------------------------------
# 9. CONSOLIDATED BENCHMARK SUMMARY & STRATEGIC COMPARISON
# ------------------------------------------------------------------------------
# Comparing how each algorithm reshapes the underlying dataset for downstream classifiers
benchmark_summary <- data.frame(
  Method = c("Original Baseline", "ROSE Under-sampling", "ROSE Over-sampling", "Condensed NN (CNN)", "SMOTE"),
  Total_Observations = c(nrow(cart_numeric), nrow(df_under), nrow(df_over), nrow(df_cnn), nrow(df_smote)),
  Majority_Count_0   = c(sum(cart_numeric$retrasos == 0), sum(df_under$retrasos == 0), sum(df_over$retrasos == 0), sum(df_cnn$retrasos == 0), sum(df_smote$retrasos == 0)),
  Minority_Count_1   = c(sum(cart_numeric$retrasos == 1), sum(df_under$retrasos == 1), sum(df_over$retrasos == 1), sum(df_cnn$retrasos == 1), sum(df_smote$retrasos == 1)),
  Minority_Percentage = c(
    mean(cart_numeric$retrasos == 1) * 100,
    mean(df_under$retrasos == 1) * 100,
    mean(df_over$retrasos == 1) * 100,
    mean(df_cnn$retrasos == 1) * 100,
    mean(df_smote$retrasos == 1) * 100
  )
)

cat("\n=== FINAL COMPARATIVE BENCHMARK: IMBALANCE REMEDIATION ===\n")
print(benchmark_summary)
