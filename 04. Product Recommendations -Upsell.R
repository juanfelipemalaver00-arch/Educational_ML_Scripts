# ------------------------------------------------------------------------------
# 1. DATASET DESCRIPTION & BUSINESS DOMAIN CONTEXT
# ------------------------------------------------------------------------------
# Dataset File: "cosmeticspan.xlsx" (Alternative: "APRIORI.xlsx")
# Domain: Cosmetics Retail & E-Commerce Cross-Selling / Upselling Analytics.
#
# Business Stake & Commercial Objectives:
#   A cosmetics brand captures basket-level Point-of-Sale (POS) transactions.
#   Each row is a consumer shopping basket; columns are binary indicator flags 
#   (1 = Purchased, 0 = Not Purchased) across product categories:
#     - `Cosmetiquera` (Cosmetic Bag / Travel Pouch)
#     - `Rubor` (Blush / Rouge)
#     - `Labial` (Lipstick)
#     - `Pestanilena` (Mascara / Eyelash Enhancer)
#     - `Base` (Liquid/Powder Foundation)
#     - `Sombra` (Eyeshadow), `Delineador` (Eyeliner), etc.
#
# Strategic Levers:
#   1. Cross-Selling & Bundling: Package high-lift items together with marginal promotional discounts.
#   2. Digital Recommendation Engines: Display real-time "Frequently Bought Together" prompts.
#   3. Planogram Optimization: Place associated products on adjacent physical shelves to boost impulse buys.
# ==============================================================================

# ------------------------------------------------------------------------------
# 2. ENVIRONMENT SETUP & DEPENDENCY MANAGEMENT
# ------------------------------------------------------------------------------
# We load pacman to dynamically manage all CRAN dependencies
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}

pacman::p_load(
  readxl,       # Fast reading of Excel sheets
  arules,       # Mining Association Rules and Frequent Itemsets (Agrawal & Srikant, 1994)
  arulesViz,    # Interactive and static network/matrix rule visualizers
  gmodels,      # Cross-tabulation and contingency tables with Pearson residuals
  vcd,          # Visualizing Categorical Data (Mosaic plots & association metrics)
  dplyr,        # Data manipulation
  ggplot2       # Plotting
)

# ------------------------------------------------------------------------------
# 3. GLOBAL CONFIGURATION & HYPERPARAMETERS
# ------------------------------------------------------------------------------
CONFIG <- list(
  # File paths
  DATA_PATH         = "cosmeticspan.xlsx",
  RULES_OUTPUT_CSV  = "recommended_rules.csv",
  
  # Visualization parameters
  TOP_N_ITEMS       = 10,
  
  # Apriori Hyperparameters:
  #   - 'MIN_SUPPORT' = 0.05: An itemset must appear in at least 5% of all transactions.
  #     Prevents the algorithm from chasing rare, unrepresentative coincidences.
  #   - 'MIN_CONFIDENCE' = 0.60: At least 60% of customers who bought X also bought Y.
  #   - 'MIN_LIFT' = 1.20: Co-occurrence must be at least 20% higher than expected by random chance.
  #   - 'MAX_LENGTH' = 4: Limits rules to 4 items (e.g., A + B + C -> D) to ensure operational practicality.
  MIN_SUPPORT       = 0.05,
  MIN_CONFIDENCE    = 0.60,
  MIN_LIFT          = 1.20,
  MAX_LENGTH        = 4,
  
  # Filtered Subrule Parameters
  HIGH_CONFIDENCE   = 0.70
)

# ------------------------------------------------------------------------------
# 4. DATA INGESTION & SPARSE TRANSACTION ENCODING
# ------------------------------------------------------------------------------
if (!file.exists(CONFIG$DATA_PATH)) {
  stop(sprintf("Fatal Error: Dataset '%s' not found.", CONFIG$DATA_PATH))
}

raw_cosmetics <- read_excel(CONFIG$DATA_PATH)
cat(sprintf("Loaded %d transactions across %d cosmetic product lines.\n", 
            nrow(raw_cosmetics), ncol(raw_cosmetics)))

# ------------------------------------------------------------------------------
# Mathematical Concept: Sparse Binary Matrix Representation (ngCMatrix)
# ------------------------------------------------------------------------------
# Why 'as(matrix, "transactions")'?
# In large retail datasets with 100,000+ SKUs, a standard dense table allocates gigabytes of RAM to zeros.
# 'arules' converts binary matrices into Compressed Sparse Column (CSC) format ('ngCMatrix'), 
# storing ONLY non-zero row indices, cutting memory usage by over 90% and accelerating bitwise operations.
cosmetics_mat <- as.matrix(raw_cosmetics)
trans_data    <- as(cosmetics_mat, "transactions")

cat("\n=== TRANSACTION INCIDENCE MATRIX SUMMARY ===\n")
summary(trans_data)

# ------------------------------------------------------------------------------
# 5. EXPLORATORY BASKET ANALYSIS: ITEM FREQUENCY PROFILING
# ------------------------------------------------------------------------------
# Function Argument Breakdown:
#   - 'topN = 10': Displays the 10 most frequent items.
#   - 'type = "relative"': Plots Relative Support P(Item) = Count(Item) / N instead of absolute counts.
itemFrequencyPlot(
  trans_data,
  topN       = CONFIG$TOP_N_ITEMS,
  type       = "relative",
  col        = "#2b83ba",
  main       = "Pareto Item Frequency: Top Cosmetic Products",
  xlab       = "Cosmetics Category",
  ylab       = "Relative Support (Purchase Proportion)",
  cex.names  = 0.8
)

# ------------------------------------------------------------------------------
# 6. STATISTICAL INDEPENDENCE AUDIT: CONTINGENCY TABLES & MOSAIC PLOTS
# ------------------------------------------------------------------------------
# Before mining complex rules, we test bivariate statistical independence between 
# two core products: 'Cosmetiquera' and 'Rubor'.
#
# Hypothesis Testing:
#   H0: Purchase of Cosmetic Bag is statistically independent of Blush purchase.
#   H1: Cosmetic Bag and Blush exhibit non-random purchase association.
#
# Mathematical Foundation of Pearson Residuals:
#   r_ij = (Observed_ij - Expected_ij) / sqrt(Expected_ij)
#   where Expected_ij = (Row_Total_i * Col_Total_j) / Grand_Total.
#   - If |r_ij| > 2: Significant association at alpha = 0.05.
#   - If |r_ij| > 4: Extremely strong association at alpha = 0.0001.
#
# Function Argument Breakdown:
#   - 'chisq = TRUE': Computes Pearson Chi-Square test of independence.
#   - 'expected = TRUE': Computes theoretical frequencies under independence.
#   - 'sresid = TRUE': Standardized Pearson residuals to identify direction of association.
cat("\n=== 2x2 CONTINGENCY TABLE & CHI-SQUARE INDEPENDENCE TEST ===\n")
cross_tab <- CrossTable(
  raw_cosmetics$Cosmetiquera,
  raw_cosmetics$Rubor,
  chisq    = TRUE,
  expected = TRUE,
  sresid   = TRUE,
  prop.r   = TRUE,
  prop.c   = TRUE
)

# Visualizing Association via Mosaic Plot (Friendly, 1994)
# Function Argument Breakdown:
#   - 'shade = TRUE': Color-codes mosaic tiles based on Pearson residuals:
#     Deep Blue = Positive association (observed >> expected).
#     Deep Red  = Negative repulsion (observed << expected).
mosaic(
  ~ Cosmetiquera + Rubor,
  data   = raw_cosmetics,
  legend = TRUE,
  shade  = TRUE,
  main   = "Mosaic Plot: Cosmetic Bag vs. Blush Interaction (Residual Shading)"
)

# ------------------------------------------------------------------------------
# 7. ASSOCIATION RULE MINING: APRIORI ALGORITHM
# ------------------------------------------------------------------------------
# Mathematical Foundations of Association Metrics:
#   Given a rule: Antecedent (X) -> Consequent (Y)
#
#   1. Support(X -> Y): Joint probability of observing both X and Y in a basket.
#      Support(X -> Y) = P(X union Y) = Count(X union Y) / N_total
#
#   2. Confidence(X -> Y): Conditional probability that Y is bought given that X is bought.
#      Confidence(X -> Y) = P(Y | X) = Support(X union Y) / Support(X)
#      Business Meaning: Direct hit-rate of the cross-sell recommendation.
#
#   3. Lift(X -> Y): Factor by which the co-occurrence exceeds random chance independence.
#      Lift(X -> Y) = P(Y | X) / P(Y) = Support(X union Y) / [ Support(X) * Support(Y) ]
#      - Lift = 1: X and Y are independent (no real cross-selling affinity).
#      - Lift > 1: Strong complementary affinity (buying X accelerates buying Y).
#      - Lift < 1: Substitutive repulsion (buying X cannibalizes buying Y).
#
#   4. Conviction(X -> Y): Measure of rule implication correctness.
#      Conviction(X -> Y) = [ 1 - P(Y) ] / [ 1 - Confidence(X -> Y) ]
#
# Mathematical Pruning Principle: Downward-Closure (Anti-Monotonicity of Support)
#   For all itemsets X, Y: If X subseteq Y, then Support(Y) <= Support(X).
#   If any sub-itemset has Support < minsup, ALL of its supersets are immediately pruned,
#   reducing worst-case search space from O(2^D) down to polynomial tractable time.
#
# Function Argument Breakdown:
#   - 'target = "rules"': Generates directional rules (X -> Y) rather than undirected frequent itemsets.
#   - 'maxlen = 4': Maximum number of items in a rule to prevent overly complex bundles.
rules <- apriori(
  trans_data,
  parameter = list(
    supp    = CONFIG$MIN_SUPPORT,
    conf    = CONFIG$MIN_CONFIDENCE,
    maxlen  = CONFIG$MAX_LENGTH,
    target  = "rules"
  )
)

cat(sprintf("\nTotal Apriori Association Rules Mined: %d\n", length(rules)))

# Sort rules by Lift descending to highlight highest-affinity product pairings
rules_sorted_lift <- sort(rules, by = "lift", decreasing = TRUE)

cat("\n=== TOP 10 HIGHEST LIFT ASSOCIATION RULES ===\n")
inspect(head(rules_sorted_lift, 10))

# ------------------------------------------------------------------------------
# 8. RULE PRUNING & STRATEGIC FILTERING
# ------------------------------------------------------------------------------
# Filter for ultra-high confidence rules (Confidence >= 70%) for automated e-commerce triggers
high_conf_rules <- subset(
  rules, 
  subset = confidence >= CONFIG$HIGH_CONFIDENCE & lift >= CONFIG$MIN_LIFT
)

cat(sprintf("\nHigh-Confidence Rules (Conf >= %.2f, Lift >= %.2f): %d\n", 
            CONFIG$HIGH_CONFIDENCE, CONFIG$MIN_LIFT, length(high_conf_rules)))
inspect(head(sort(high_conf_rules, by = "lift"), 10))

# ------------------------------------------------------------------------------
# 9. PUBLICATION-GRADE RULE VISUALIZATIONS (ARULESVIZ)
# ------------------------------------------------------------------------------
# A. Support vs. Lift Scatter Plot colored by Confidence
# Function Argument Breakdown:
#   - 'measure = c("support", "lift")': Defines X and Y axes.
#   - 'shading = "confidence"': Color gradient maps rule confidence.
plot(
  rules,
  measure = c("support", "lift"),
  shading = "confidence",
  main    = "Association Rules: Support vs. Lift Landscape",
  col     = heat.colors(10)
)

# B. Matrix Layout of Filtered Rules
# Function Argument Breakdown:
#   - 'method = "matrix"': Plots Antecedents on rows and Consequents on columns.
#   - 'control = list(reorder = "support/confidence")': Clusters similar products together.
if (length(high_conf_rules) > 1) {
  plot(
    head(sort(high_conf_rules, by = "lift"), 15),
    method  = "matrix",
    measure = "lift",
    control = list(reorder = "support/confidence")
  )
}

# C. Network Graph Representation
# Function Argument Breakdown:
#   - 'method = "graph"': Visualizes rules as directed network vertices and edges.
#   - 'control = list(type = "items")': Product items serve as graph nodes; circles represent rule vertices.
top_graph_rules <- head(rules_sorted_lift, 10)
plot(
  top_graph_rules,
  method  = "graph",
  control = list(type = "items")
)

# ------------------------------------------------------------------------------
# 10. DEPLOYMENT & COMMERCE INTEGRATION EXPORT
# ------------------------------------------------------------------------------
# Export clean tabular rules into CSV for seamless ingestion by E-Commerce 
# recommendation microservices or ERP bundling systems.
rules_df <- as(rules_sorted_lift, "data.frame")
write.csv2(rules_df, file = CONFIG$RULES_OUTPUT_CSV, row.names = FALSE)

cat(sprintf("\nSuccessfully exported %d production rules to '%s'.\n", 
            nrow(rules_df), CONFIG$RULES_OUTPUT_CSV))
