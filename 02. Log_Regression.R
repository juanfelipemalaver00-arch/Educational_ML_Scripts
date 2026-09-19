# ------------------------------------------------------------------------------
# 1. DATASET DESCRIPTIONS & BUSINESS CONTEXT
# ------------------------------------------------------------------------------
# Part A - Retail Banking Customer Satisfaction ("financierosatisfaccion.xlsx"):
#   - Domain: Retail Banking Customer Experience Management.
#   - Features: `edad` (Age in years) and `saldo` (Account balance in USD/COP).
#   - Target: `satisfecho` (1 = Satisfied, 0 = Dissatisfied).
#   - Business Rationale: Identify which demographic/financial levers drive customer
#     advocacy and prevent account churn.
#
# Part B - Enterprise Credit Risk ("carteraguia2017.xlsx"):
#   - Domain: B2B Healthcare Accounts Receivable Portfolio.
#   - Target: `retrasos` (1 = Delinquency/Default, 0 = On-time payment).
#   - Financial Problem: Setting arbitrary classification cutoffs (like 0.50) ignores 
#     asymmetric financial costs: False Negatives (undetected defaults) cost thousands, 
#     while False Positives (auditing good clients) cost only minimal staff time.
#   - Analytical Solution: Construct a dynamic Cost-Benefit Profit Curve to find the 
#     exact probability threshold that maximizes net dollar recovery.
# ==============================================================================

# ------------------------------------------------------------------------------
# 2. ENVIRONMENT SETUP & DEPENDENCY MANAGEMENT
# ------------------------------------------------------------------------------
# We load 'pacman' to streamline package deployment
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}

pacman::p_load(
  readxl,       # Fast reading of Excel sheets
  dplyr,        # Tidy data manipulation
  tibble,       # Modern, robust data frames
  caret,        # Model training, stratified partitioning, and confusion matrices
  ROCR,         # Receiver Operating Characteristic and Performance visualizer
  ggplot2,      # Data visualization
  scales        # Number and currency formatting
)

# ------------------------------------------------------------------------------
# 3. GLOBAL CONFIGURATION & HYPERPARAMETERS
# ------------------------------------------------------------------------------
CONFIG <- list(
  # File paths
  PATH_FINANCIAL = "financierosatisfaccion.xlsx",
  PATH_CARTERA   = "carteraguia2017.xlsx",
  
  # Reproducibility Seeds
  SEED_SPLIT     = 1545867,
  SEED_BALANCED  = 24029114,
  
  # Train/Test Partitioning
  # Why 75/25 split? Provides sufficient statistical power for training (N_train) 
  # while leaving an unbiased holdout partition (N_test) to detect overfitting.
  TRAIN_RATIO    = 0.75,
  
  # Standard baseline classification cutoff
  DEFAULT_CUTOFF = 0.50,
  
  # Asymmetric Business Cost-Benefit Matrix ($ per account):
  #   - BENEFIT_RECOVERED: Average gross debt saved if a delinquent account is identified ($500)
  #   - COST_INTERVENTION: Administrative, legal, and audit cost to investigate a flagged account ($100)
  BENEFIT_RECOVERED_PER_CASE = 500,
  COST_INTERVENTION_PER_CASE = 100,
  
  THEME_BASE_SIZE = 12
)

set.seed(CONFIG$SEED_SPLIT)

# ==============================================================================
# PART A: MATHEMATICAL FOUNDATIONS OF LOGISTIC REGRESSION & ODDS RATIOS
# ==============================================================================
cat("\n=======================================================\n")
cat("PART A: INTRODUCTORY LOGISTIC REGRESSION & ODDS RATIOS\n")
cat("=======================================================\n")

# Ingest or fallback to embedded dataset
if (file.exists(CONFIG$PATH_FINANCIAL)) {
  df_satis <- read_excel(CONFIG$PATH_FINANCIAL)
} else {
  df_satis <- tibble(
    edad = c(30, 33, 35, 30, 59, 35, 36, 39, 41, 43, 39, 43),
    saldo = c(1787, 4108, 1350, 1476, 0, 747, 307, 147, 221, -88, 3374, 264),
    satisfecho = c(1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0)
  )
}

df_satis <- df_satis %>% mutate(satisfecho = as.factor(satisfecho))

# ------------------------------------------------------------------------------
# Mathematical Concept: Generalized Linear Models (GLM) & The Logit Link
# ------------------------------------------------------------------------------
# Why not Ordinary Least Squares (OLS)?
# Linear regression produces predictions (-inf, +inf) violating the probability bounds [0, 1].
# Logistic regression transforms probabilities p in [0, 1] into real space (-inf, +inf) via the logit link:
#   eta_i = ln( p_i / (1 - p_i) ) = beta_0 + beta_1 * edad_i + beta_2 * saldo_i
#
# Optimization Criterion:
#   Parameters beta are estimated via Maximum Likelihood Estimation (MLE) using
#   Iteratively Reweighted Least Squares (IRLS), maximizing the log-likelihood:
#   ln L(beta) = sum_{i=1}^N [ y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) ]
#
# Function Argument Breakdown:
#   - 'family = binomial(link = "logit")': Declares the Bernoulli/Binomial error structure 
#     and the canonical log-odds link function.

logit_satis <- glm(satisfecho ~ edad + saldo, family = binomial(link = "logit"), data = df_satis)
summary(logit_satis)

# ------------------------------------------------------------------------------
# Mathematical Concept: Odds Ratios (OR)
# ------------------------------------------------------------------------------
# The coefficient beta_j represents the change in *log-odds* for a 1-unit increase in X_j.
# Exponentiating yields the Odds Ratio:
#   OR_j = exp(beta_j) = Odds(X_j + 1) / Odds(X_j)
#   - If OR > 1: Predictor increases the odds of the event (positive driver).
#   - If OR < 1: Predictor decreases the odds of the event (protective factor).
#   - If OR = 1: Predictor has zero relationship with the event.
odds_ratios_satis <- exp(coef(logit_satis))
cat("\n--- Odds Ratios (Multiplicative Odds Factor) ---\n")
print(round(odds_ratios_satis, 4))

# ------------------------------------------------------------------------------
# Mathematical Concept: Inverse Logit (Sigmoid) Prediction
# ------------------------------------------------------------------------------
# Function Argument Breakdown:
#   - 'type = "response"': Computes the inverse link function to return the actual probability p:
#     p = 1 / (1 + exp(-eta)) = 1 / (1 + exp(-(beta_0 + beta_1*edad + beta_2*saldo)))
#   - (Note: 'type = "link"' would return the raw linear log-odds eta).
new_customer <- data.frame(edad = 41, saldo = 1000)
pred_prob_satis <- predict(logit_satis, newdata = new_customer, type = "response")
cat(sprintf("\nPredicted Probability of Satisfaction for (Age=41, Balance=$1000): %.2f%%\n", pred_prob_satis * 100))


# ==============================================================================
# PART B: ENTERPRISE CREDIT RISK, STEPWISE AIC & PROFIT OPTIMIZATION
# ==============================================================================
cat("\n=======================================================\n")
cat("PART B: B2B CREDIT RISK & STEPWISE FEATURE SELECTION\n")
cat("=======================================================\n")

if (!file.exists(CONFIG$PATH_CARTERA)) {
  stop(sprintf("Fatal Error: Dataset '%s' not found.", CONFIG$PATH_CARTERA))
}

cartera_raw <- read_excel(CONFIG$PATH_CARTERA)

# Data Preprocessing & Categorical Transformation
cartera_prepped <- cartera_raw %>%
  mutate(
    TIPOips = as.factor(TIPO_ips),
    retrasos = as.factor(retrasos)
  )

# One-Hot Encoding (Dummy Expansion)
# Why 'dummyVars'? Converts multi-level categorical factors into orthogonal binary (0/1) indicator columns.
# Argument 'fullRank = FALSE': Keeps all levels for full inspection before manual multicollinearity pruning.
dummy_model <- dummyVars(" ~ .", data = cartera_prepped, fullRank = FALSE)
cartera_encoded <- as.data.frame(predict(dummy_model, newdata = cartera_prepped))

# Prune Collinear & Redundant Levels to Prevent the Dummy Variable Trap
cartera_clean <- cartera_encoded %>%
  select(
    -matches("TIPOips\\.5|TIPOips\\.2|TIPO_ips|VENTAS_MENS_PROMEDIO|cartera_actual_insumos_B")
  ) %>%
  mutate(retrasos = as.factor(retrasos))

# ------------------------------------------------------------------------------
# Stratified Train/Test Splitting
# ------------------------------------------------------------------------------
# Why 'createDataPartition'? Random splitting in imbalanced data can yield zero defaults in test.
# Stratified splitting guarantees that the train and test sets have the *exact same* default proportion.
set.seed(CONFIG$SEED_SPLIT)
train_idx <- createDataPartition(cartera_clean$retrasos, p = CONFIG$TRAIN_RATIO, list = FALSE)
train_set <- cartera_clean[train_idx, ]
test_set  <- cartera_clean[-train_idx, ]

cat(sprintf("Training Observations: %d | Holdout Test Observations: %d\n", nrow(train_set), nrow(test_set)))

# ------------------------------------------------------------------------------
# Stepwise Akaike Information Criterion (AIC) Model Selection
# ------------------------------------------------------------------------------
# Mathematical Foundation:
#   AIC balances model fit (Likelihood L) against model complexity (k parameters):
#   AIC = 2k - 2 * ln(L)
#   Minimizing AIC avoids both underfitting (high bias) and overfitting (high variance).
#
# Function Argument Breakdown:
#   - 'direction = "both"': Bidirectional stepwise search. At each step, it tests adding 
#     a new feature (forward entry) or dropping an existing feature (backward elimination).
#   - 'trace = 0': Suppresses printing hundreds of intermediate model outputs to keep logs clean.
full_model <- glm(retrasos ~ ., family = binomial(link = "logit"), data = train_set)
stepwise_model <- step(full_model, direction = "both", trace = 0)

cat("\n--- Final Stepwise Logistic Model Coefficients ---\n")
summary(stepwise_model)

step_coefs <- coef(stepwise_model)
step_odds  <- exp(step_coefs)
cat("\n--- Stepwise Odds Ratios ---\n")
print(round(step_odds, 4))

# ------------------------------------------------------------------------------
# Vectorized Step-Ahead Odds & Probability Trajectory
# ------------------------------------------------------------------------------
# Why do we do this?
# In credit risk committees, executives ask: "What happens to default risk if debt ratio increases by 1, 2, or 5 steps?"
# We broadcast the baseline odds across standard deviation steps vectorially using 'outer()'.
base_counts <- table(train_set$retrasos)
base_odds   <- (base_counts["1"] / sum(base_counts)) / (base_counts["0"] / sum(base_counts))
predictor_odds <- step_odds[-1] # Exclude intercept

steps <- -10:10
odds_trajectory <- outer(predictor_odds, steps, function(or, s) base_odds * (or ^ s))
prob_trajectory <- odds_trajectory / (1 + odds_trajectory)

cat(sprintf("\nBaseline Portfolio Odds: %.4f | Baseline Event Probability: %.2f%%\n", 
            base_odds, (base_odds / (1 + base_odds)) * 100))

# ------------------------------------------------------------------------------
# Out-of-Sample Model Evaluation (Confusion Matrix at Default 0.50 Threshold)
# ------------------------------------------------------------------------------
# Predict probabilities on unseen holdout test set
test_probs <- predict(stepwise_model, newdata = test_set, type = "response")
test_preds <- factor(ifelse(test_probs >= CONFIG$DEFAULT_CUTOFF, "1", "0"), levels = c("0", "1"))

# Function Argument Breakdown:
#   - 'positive = "1"': Crucial! Specifies that class '1' (Delinquency) is the positive event.
#     Without this, R may calculate Sensitivity for class '0' (non-defaults), misleading the business!
cm_test <- confusionMatrix(test_preds, test_set$retrasos, positive = "1")
cat("\n=== HOLDOUT CONFUSION MATRIX (Cutoff = 0.50) ===\n")
print(cm_test$table)
print(round(cm_test$byClass, 4))

# ------------------------------------------------------------------------------
# Receiver Operating Characteristic (ROC) & AUC
# ------------------------------------------------------------------------------
# Mathematical Foundation:
#   - True Positive Rate (TPR / Recall / Sensitivity): TPR = TP / (TP + FN) = P(Y_hat=1 | Y=1)
#   - False Positive Rate (FPR / Fall-out): FPR = FP / (FP + TN) = P(Y_hat=1 | Y=0)
#   - Area Under the Curve (AUC): Probability that a randomly chosen defaulting account 
#     receives a higher predicted risk score than a randomly chosen non-defaulting account:
#     AUC = P(Score(Default) > Score(Non-Default)) in [0.5 (random), 1.0 (perfect)].
pred_obj <- prediction(test_probs, test_set$retrasos)
roc_perf <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
auc_perf <- performance(pred_obj, measure = "auc")
auc_value <- auc_perf@y.values[[1]]

cat(sprintf("\nArea Under the ROC Curve (AUC): %.4f\n", auc_value))

# Publication-grade ROC visualization
roc_df <- data.frame(
  FPR = unlist(roc_perf@x.values),
  TPR = unlist(roc_perf@y.values),
  Cutoff = unlist(roc_perf@alpha.values)
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "#1f77b4", size = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  annotate("text", x = 0.6, y = 0.2, label = sprintf("AUC = %.3f", auc_value), size = 5, fontface = "bold") +
  labs(
    title = "Receiver Operating Characteristic (ROC) Curve",
    subtitle = "Healthcare B2B Default Risk Model (Discrimination Performance)",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Recall / Sensitivity)"
  ) +
  theme_minimal(base_size = CONFIG$THEME_BASE_SIZE)

print(p_roc)

# ------------------------------------------------------------------------------
# STRATEGIC DATA SCIENCE: COST-BENEFIT PROFIT OPTIMIZATION CURVE
# ------------------------------------------------------------------------------
# Why do we do this?
# In credit risk, threshold 0.50 is arbitrary and sub-optimal.
# A False Negative (unidentified defaulting hospital) causes a default loss of $500.
# A False Positive (investigating a healthy hospital) costs $100 in auditing costs.
#
# Mathematical Formulation of Expected Net Profit per Account as a Function of Cutoff c:
#   Net_Profit(c) = P(Default) * Recall(c) * [ Benefit_Recovered - ( Cost_Intervention / Precision(c) ) ]
#
# Derivation:
#   - P(Default) * Recall(c) = True Positives identified per account.
#   - For each True Positive, the company gains $Benefit_Recovered ($500).
#   - But to catch each True Positive, the company audits (1 / Precision(c)) accounts,
#     each costing $Cost_Intervention ($100).

rec_perf  <- performance(pred_obj, measure = "rec")
prec_perf <- performance(pred_obj, measure = "prec")

cutoffs    <- unlist(rec_perf@x.values)
recalls    <- unlist(rec_perf@y.values)
precisions <- unlist(prec_perf@y.values)

prior_p <- mean(test_set$retrasos == "1")

profit_df <- tibble(
  Cutoff = cutoffs,
  Recall = recalls,
  Precision = precisions
) %>%
  filter(!is.na(Precision) & Precision > 0 & is.finite(Cutoff)) %>%
  mutate(
    Net_Profit = prior_p * Recall * (CONFIG$BENEFIT_RECOVERED_PER_CASE - (CONFIG$COST_INTERVENTION_PER_CASE / Precision))
  )

# Find the profit-maximizing probability threshold
optimal_row    <- profit_df %>% slice_max(Net_Profit, n = 1)
optimal_cutoff <- optimal_row$Cutoff[1]
max_profit     <- optimal_row$Net_Profit[1]

cat(sprintf("\n=== OPTIMAL FINANCIAL DECISION THRESHOLD ===\n"))
cat(sprintf("Optimal Probability Cutoff: %.4f\n", optimal_cutoff))
cat(sprintf("Max Expected Net Profit per Portfolio Account: $%.2f\n", max_profit))
cat(sprintf("At this cutoff -> Precision: %.2f%% | Recall: %.2f%%\n", 
            optimal_row$Precision[1] * 100, optimal_row$Recall[1] * 100))

# Plot the Financial Profit Curve
p_profit <- ggplot(profit_df, aes(x = Cutoff, y = Net_Profit)) +
  geom_line(color = "#2ca02c", size = 1.2) +
  geom_vline(xintercept = optimal_cutoff, linetype = "dashed", color = "darkred", size = 1) +
  annotate("text", x = optimal_cutoff, y = max_profit * 0.9, 
           label = sprintf("Optimal Cutoff = %.2f\nMax Profit = $%.2f/acct", optimal_cutoff, max_profit),
           hjust = -0.1, color = "darkred", fontface = "bold") +
  labs(
    title = "Financial Cost-Benefit Threshold Optimization",
    subtitle = "Connecting Classifier Probabilities directly to Net Cash Recovery",
    x = "Probability Threshold (Cutoff)",
    y = "Expected Net Economic Return ($ per portfolio account)"
  ) +
  theme_minimal(base_size = CONFIG$THEME_BASE_SIZE)

print(p_profit)
