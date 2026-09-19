# ------------------------------------------------------------------------------
# 1. DATASET DESCRIPTION & BUSINESS CONTEXT
# ------------------------------------------------------------------------------
# Dataset Files: "cerealtrain.csv" (Training Partition) & "cerealtest.csv" (Test Partition)
# Domain: Fast-Moving Consumer Goods (FMCG) / Food & Beverage Marketing Analytics.
#
# Business Reality & Strategic Objective:
#   A global breakfast cereal conglomerate wants to optimize product line architecture 
#   and targeted advertising campaigns. To allocate marketing budget across brands 
#   (e.g., Healthy/Oat, Sugary/Kids, High-Protein, Traditional Corn), the business 
#   must accurately predict breakfast product choice based on demographic segments.
#
# Features:
#   - `edadcat`: Consumer age category (e.g., Teens, Young Adults, Mature, Seniors).
#   - `genero`: Gender (Male / Female).
#   - `ecivil`: Marital status (Single, Married, Divorced, Widowed).
#   - `activo`: Physical activity lifestyle index (High, Moderate, Sedentary).
#   - `desayuno` (Multi-class Target): Breakfast category consumed.
#
# Pedagogical Question:
#   How do we prevent individual decision trees from memorizing training noise (high variance), 
#   and how does ensemble averaging in Random Forests mathematically reduce generalization error?
# ==============================================================================

# ------------------------------------------------------------------------------
# 2. ENVIRONMENT SETUP & DEPENDENCY MANAGEMENT
# ------------------------------------------------------------------------------
# We load pacman to dynamically manage all CRAN dependencies
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}

pacman::p_load(
  rpart,        # Recursive Partitioning and Regression Trees (Breiman et al. CART)
  rpart.plot,   # High-resolution, professional tree plotting
  party,        # Conditional Inference Trees (ctree - permutation test splitting)
  mvtnorm,      # Multivariate normal distributions (party dependency)
  caret,        # Multi-class confusion matrices and cross-validation harnesses
  randomForest, # Classical Breiman & Cutler Random Forest engine
  ranger,       # Fast, multi-threaded C++ Random Forest implementation
  ggplot2,      # Visual analytics
  dplyr,        # Tidy data wrangling
  scales        # Axis formatting
)

# ------------------------------------------------------------------------------
# 3. GLOBAL CONFIGURATION & HYPERPARAMETERS
# ------------------------------------------------------------------------------
CONFIG <- list(
  # Data paths
  PATH_TRAIN       = "cerealtrain.csv",
  PATH_TEST        = "cerealtest.csv",
  
  # Reproducibility Seed
  SEED_GLOBAL      = 123,
  
  # CART Hyperparameters:
  #   - 'MIN_SPLIT' = 20: Minimum observations required in a node to attempt a split.
  #   - 'MIN_BUCKET' = 7: Minimum terminal leaf size. Prevents single-instance leaves.
  #   - 'CP_OPTIMAL' = 0.02: Complexity penalty parameter for tree pruning.
  MIN_SPLIT        = 20,
  MIN_BUCKET       = 7,
  CP_OPTIMAL       = 0.02,
  
  # Conditional Inference Tree (CTree) Parameters:
  #   - 'CTREE_MIN_CRIT' = 0.95: Corresponds to (1 - alpha) where alpha = 0.05.
  #     Splits only occur if the permutation test p-value < 0.05.
  #   - 'CTREE_MAX_DEPTH' = 5: Maximum tree depth to guarantee human interpretability.
  CTREE_MIN_CRIT   = 0.95,
  CTREE_MAX_DEPTH  = 5,
  
  # Random Forest Hyperparameters:
  #   - 'RF_NTREE' = 500: Number of bootstrap trees grown. As B -> inf, variance -> 0.
  #   - 'RF_MTRY_DEFAULT' = 2: Number of features randomly sampled at each split (approx sqrt(p)).
  RF_NTREE         = 500,
  RF_MTRY_DEFAULT  = 2,
  
  # Fast Grid Search (Ranger + Caret)
  CV_FOLDS         = 5,
  TUNE_GRID        = expand.grid(
    mtry          = c(1, 2, 3, 4),
    splitrule     = "gini",
    min.node.size = c(1, 5, 10)
  ),
  
  THEME_BASE_SIZE  = 12
)

set.seed(CONFIG$SEED_GLOBAL)

# ------------------------------------------------------------------------------
# 4. DATA INGESTION & FACTOR ENCODING
# ------------------------------------------------------------------------------
if (!file.exists(CONFIG$PATH_TRAIN) || !file.exists(CONFIG$PATH_TEST)) {
  stop("Fatal Error: Training or Test cereal datasets are missing.")
}

# Ingest CSVs with semicolon delimiter and comma decimals
train_raw <- read.csv(CONFIG$PATH_TRAIN, header = TRUE, sep = ";", dec = ",")
test_raw  <- read.csv(CONFIG$PATH_TEST, header = TRUE, sep = ";", dec = ",")

# Vectorized Factor Conversion
# Why factor typing? Tree algorithms require discrete variables to be typed as factors 
# to evaluate categorical subset partitionings (e.g., {Single, Divorced} vs {Married}).
clean_cereal_data <- function(df) {
  df %>%
    mutate(across(where(is.character), as.factor)) %>%
    mutate(desayuno = as.factor(desayuno))
}

cerealtrain <- clean_cereal_data(train_raw)
cerealtest  <- clean_cereal_data(test_raw)

cat(sprintf("Training Sample Size: %d | Holdout Test Sample Size: %d\n", nrow(cerealtrain), nrow(cerealtest)))
cat("Target Categories:", paste(levels(cerealtrain$desayuno), collapse = ", "), "\n")

# ------------------------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
plot_categorical_feature <- function(data, var_col, fill_color, title_text) {
  ggplot(data, aes(x = .data[[var_col]])) +
    geom_bar(fill = fill_color, alpha = 0.85, color = "black") +
    geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5, fontface = "bold") +
    labs(
      title = title_text,
      x = var_col,
      y = "Consumer Count"
    ) +
    theme_minimal(base_size = CONFIG$THEME_BASE_SIZE)
}

p1 <- plot_categorical_feature(cerealtrain, "edadcat",  "#e66101", "Age Segment Breakdown")
p2 <- plot_categorical_feature(cerealtrain, "genero",   "#5e3c99", "Gender Breakdown")
p3 <- plot_categorical_feature(cerealtrain, "ecivil",   "#2b83ba", "Marital Status Breakdown")
p4 <- plot_categorical_feature(cerealtrain, "desayuno", "#d7191c", "Target: Breakfast Product Choice")

print(p1)
print(p2)
print(p3)
print(p4)

# ------------------------------------------------------------------------------
# 6. MODEL 1: CART DECISION TREE & COST-COMPLEXITY PRUNING
# ------------------------------------------------------------------------------
# Mathematical Foundation: Recursive Binary Splitting
# At each node t, CART searches across all features j and all split thresholds s to maximize Gini Gain:
#   Delta I_G(s, t) = I_G(t) - [ (N_L / N_t) * I_G(t_L) + (N_R / N_t) * I_G(t_R) ]
# where Gini Impurity is:
#   I_G(t) = 1 - sum_{k=1}^K p(k | t)^2
#
# Function Argument Breakdown:
#   - 'method = "class"': Fits a categorical classification tree (as opposed to "anova" for regression).
#   - 'control = rpart.control(cp = 0.001, minsplit = 10, minbucket = 3)':
#     Grows a deliberately deep, overfitted tree with a tiny complexity parameter (cp = 0.001) 
#     so we can subsequently analyze where to prune.
tree_base <- rpart(
  desayuno ~ .,
  data    = cerealtrain,
  method  = "class",
  control = rpart.control(cp = 0.001, minsplit = 10, minbucket = 3)
)

# ------------------------------------------------------------------------------
# Cost-Complexity Pruning Theory (Breiman et al., 1984)
# ------------------------------------------------------------------------------
# An unpruned tree achieves zero training error but high test variance.
# We define the Cost-Complexity Objective Function:
#   R_alpha(T) = R(T) + alpha * |T|
# where:
#   - R(T) is the misclassification cost of subtree T.
#   - |T| is the number of terminal leaves in subtree T.
#   - alpha (represented by CP in rpart) is the complexity penalty per leaf.
#
# How to read the CP Table ('printcp'):
#   - 'CP': Complexity parameter threshold.
#   - 'nsplit': Number of splits in the tree (|T| = nsplit + 1).
#   - 'rel error': Empirical training error relative to root node (1.0).
#   - 'xerror': 10-fold cross-validation error.
#   - 'xstd': Standard error of cross-validation error.
#
# 1-SE Rule: Choose the simplest tree (smallest nsplit) whose xerror is within 
# 1 standard deviation (xstd) of the minimal xerror.
cat("\n=== CART COST-COMPLEXITY (CP) TABLE ===\n")
printcp(tree_base)
plotcp(tree_base)

# Prune the tree using the optimal CP parameter
tree_pruned <- prune(tree_base, cp = CONFIG$CP_OPTIMAL)

# ------------------------------------------------------------------------------
# Professional CART Visualization
# ------------------------------------------------------------------------------
# Function Argument Breakdown:
#   - 'type = 4': Labels split branches with variable conditions for intuitive reading.
#   - 'extra = 104': Displays the predicted class, the probability distribution across 
#     all classes, and the percentage of total observations that fall into that node.
#   - 'under = TRUE': Puts node statistics underneath the box.
#   - 'fallen.leaves = TRUE': Aligns all terminal leaf nodes at the bottom margin.
#   - 'box.palette = "GnBu"': Applies a gradient palette from Green to Blue based on node purity.
rpart.plot(
  tree_pruned,
  type         = 4,
  extra        = 104,
  under        = TRUE,
  fallen.leaves= TRUE,
  box.palette  = "GnBu",
  shadow.col   = "gray80",
  main         = "Optimized CART Decision Tree (Pruned)"
)

# ------------------------------------------------------------------------------
# 7. MODEL 2: CONDITIONAL INFERENCE TREE (CTREE)
# ------------------------------------------------------------------------------
# Why do we do this? (Addressing CART's Fundamental Selection Bias)
# CART has a known mathematical flaw: it is biased towards selecting features with 
# many categories or continuous distributions because they offer more split candidates.
#
# CTree Solution (Hothorn, Hornik, Zeileis, 2006):
#   Separates feature selection from split determination using non-parametric permutation tests:
#   1. Tests the null hypothesis of independence: H0: P(Y | X_j) = P(Y).
#   2. Computes the permutation p-value for each feature X_j.
#   3. Selects the feature with the lowest p-value (strongest association).
#   4. If min(p-value) > alpha (e.g., 0.05), stopping criterion is reached (NO pruning required!).
#
# Function Argument Breakdown:
#   - 'mincriterion = 0.95': Equivalent to stopping when p-value > 0.05 (1 - 0.95 = 0.05).
#   - 'minsplit = 20': Minimum sample size in node before hypothesis test is computed.
#   - 'maxdepth = 5': Limits branch depth to prevent cognitive overload for stakeholders.
tree_ctree <- ctree(
  desayuno ~ .,
  data     = cerealtrain,
  controls = ctree_control(
    mincriterion = CONFIG$CTREE_MIN_CRIT,
    minsplit     = CONFIG$MIN_SPLIT,
    maxdepth     = CONFIG$CTREE_MAX_DEPTH
  )
)

plot(
  tree_ctree,
  main = "Conditional Inference Tree (p-value Hypothesis Splitting)",
  inner_panel = node_inner(tree_ctree, pval = TRUE, id = TRUE),
  terminal_panel = node_terminal(tree_ctree, fill = "lightblue")
)

# ------------------------------------------------------------------------------
# 8. MULTI-MODEL BENCHMARK: TRAIN VS TEST GENERALIZATION
# ------------------------------------------------------------------------------
evaluate_classifier <- function(model_obj, test_data, target_col, model_name, is_ctree = FALSE) {
  if (is_ctree) {
    preds <- predict(model_obj, newdata = test_data, type = "response")
  } else {
    preds <- predict(model_obj, newdata = test_data, type = "class")
  }
  
  cm <- confusionMatrix(preds, test_data[[target_col]])
  
  # Macro F1: unweighted arithmetic mean of F1 across all target breakfast categories
  by_class_metrics <- as.data.frame(cm$byClass)
  f1_macro <- mean(by_class_metrics$F1, na.rm = TRUE)
  accuracy <- cm$overall["Accuracy"]
  
  return(list(
    Name     = model_name,
    Accuracy = as.numeric(accuracy),
    F1_Macro = as.numeric(f1_macro),
    CM       = cm
  ))
}

eval_base_train   <- evaluate_classifier(tree_base,   cerealtrain, "desayuno", "CART Base (Train - Overfit)")
eval_base_test    <- evaluate_classifier(tree_base,   cerealtest,  "desayuno", "CART Base (Test - Holdout)")
eval_pruned_test  <- evaluate_classifier(tree_pruned, cerealtest,  "desayuno", "CART Pruned (Test - 1-SE)")
eval_ctree_test   <- evaluate_classifier(tree_ctree,  cerealtest,  "desayuno", "CTree (Test - Permutation)", is_ctree = TRUE)

tree_benchmark <- data.frame(
  Model = c(eval_base_train$Name, eval_base_test$Name, eval_pruned_test$Name, eval_ctree_test$Name),
  Accuracy_Pct = round(c(eval_base_train$Accuracy, eval_base_test$Accuracy, eval_pruned_test$Accuracy, eval_ctree_test$Accuracy) * 100, 2),
  Macro_F1_Pct = round(c(eval_base_train$F1_Macro, eval_base_test$F1_Macro, eval_pruned_test$F1_Macro, eval_ctree_test$F1_Macro) * 100, 2)
)

cat("\n=== DECISION TREE BENCHMARK (TRAIN VS TEST GENERALIZATION) ===\n")
print(tree_benchmark)

# ------------------------------------------------------------------------------
# 9. MODEL 3: ENSEMBLE LEARNING - RANDOM FOREST (BREIMAN)
# ------------------------------------------------------------------------------
# Mathematical Foundation: Bagging & Feature De-correlation
# Individual decision trees have low bias but high variance (instability).
# Random Forest trains B decorrelated trees using Bootstrap Aggregation (Bagging):
#
# Variance of the Average of B Trees:
#   Var(1/B sum_{b=1}^B T_b(x)) = rho * sigma^2 + (1 - rho) / B * sigma^2
# where:
#   - sigma^2 is the variance of a single tree.
#   - rho is the pairwise correlation between trees.
#   - As B -> inf, the second term vanishes to 0.
#   - By forcing each split to sample only 'mtry' features randomly, Random Forest
#     actively *decreases rho*, shrinking total model variance to near zero!
#
# Function Argument Breakdown:
#   - 'ntree = 500': Grows 500 bootstrap trees.
#   - 'mtry = 2': At each candidate split, randomly selects 2 predictors out of p.
#   - 'importance = TRUE': Calculates Mean Decrease Accuracy (via Out-of-Bag permutation) 
#     and Mean Decrease Gini.
set.seed(CONFIG$SEED_GLOBAL)
rf_model <- randomForest(
  desayuno ~ .,
  data       = cerealtrain,
  ntree      = CONFIG$RF_NTREE,
  mtry       = CONFIG$RF_MTRY_DEFAULT,
  importance = TRUE
)

cat("\n=== BASE RANDOM FOREST (BREIMAN) SUMMARY ===\n")
print(rf_model)

# Feature Importance Plot
# Mean Decrease Accuracy: How much out-of-bag accuracy drops when feature values are randomly shuffled.
# Mean Decrease Gini: Total Gini purity reduction contributed by splits on that variable.
varImpPlot(
  rf_model,
  main = "Random Forest: Predictor Importance Hierarchy",
  pch  = 19,
  col  = "darkblue"
)

# ------------------------------------------------------------------------------
# 10. ADVANCED CROSS-VALIDATED TUNING: FAST MULTI-THREADED RANGER + CARET
# ------------------------------------------------------------------------------
# Why 'ranger'?
# Classical 'randomForest' in R runs on a single thread and can be slow.
# 'ranger' is a high-performance C++ implementation that utilizes all CPU cores.
#
# Function Argument Breakdown:
#   - 'method = "cv", number = 5': 5-Fold Cross-Validation.
#   - 'tuneGrid = CONFIG$TUNE_GRID': Grid search evaluating mtry in {1,2,3,4} and min.node.size in {1,5,10}.
#   - 'splitrule = "gini"': Standard classification splitting criterion.
fit_control <- trainControl(
  method          = "cv",
  number          = CONFIG$CV_FOLDS,
  selectionFunction = "best"
)

set.seed(CONFIG$SEED_GLOBAL)
rf_tuned_caret <- train(
  desayuno ~ .,
  data      = cerealtrain,
  method    = "ranger",
  trControl = fit_control,
  tuneGrid  = CONFIG$TUNE_GRID,
  importance= "impurity"
)

cat("\n=== RANGER HYPERPARAMETER GRID SEARCH RESULTS ===\n")
print(rf_tuned_caret$results)
cat("Optimal Hyperparameter Combination:\n")
print(rf_tuned_caret$bestTune)

# ------------------------------------------------------------------------------
# 11. FINAL OUT-OF-SAMPLE TEST EVALUATION (TUNED ENSEMBLE)
# ------------------------------------------------------------------------------
rf_tuned_preds <- predict(rf_tuned_caret, newdata = cerealtest)
cm_rf_tuned    <- confusionMatrix(rf_tuned_preds, cerealtest$desayuno)

cat("\n=== FINAL TEST CONFUSION MATRIX (TUNED RANDOM FOREST) ===\n")
print(cm_rf_tuned$table)
cat(sprintf("Final Holdout Accuracy: %.2f%%\n", cm_rf_tuned$overall["Accuracy"] * 100))
cat(sprintf("Final Holdout Macro F1: %.2f%%\n", mean(cm_rf_tuned$byClass[, "F1"], na.rm = TRUE) * 100))
