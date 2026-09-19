# ------------------------------------------------------------------------------
# 1. DATASET DESCRIPTION & BUSINESS DOMAIN FOUNDATIONS
# ------------------------------------------------------------------------------
# Dataset File: "ASOMEDIO2012A2025MES.csv"
# Domain: Media Advertising Expenditure & Marketing Mix Modeling (Asomedios Colombia).
# Scope: Monthly investment data across all major media channels from 2012 through 2025.
#
# Tracked Advertising Channels:
#   - `TV.REG.Y.LOCAL`: Regional & Local Television broadcasts.
#   - `TV.NACIONAL`: Major National Television Networks (high fixed production and airtime costs).
#   - `RADIO`: Commercial Radio AM/FM and affiliated audio streams.
#   - `REVISTAS`: Print and Digital Magazines (niche consumer segments).
#   - `PUB.EXTERIOR`: Out-of-Home (OOH) Billboards, Transit, and LED displays.
#   - `PERIODICOS`: National and Regional Daily Newspapers.
#   - `TOTAL.CON.PERIODICOS`: Aggregate Macro-Market Advertising Volume.
#
# Core Business Problems:
#   1. Volatility & Budget Allocation: Media inflation, macro shocks (e.g., 2020 lockdowns), 
#      and structural Q4 seasonality (Black Friday / Holiday campaigns) cause severe budget over/underspend.
#   2. The Forecaster's Tournament: No single algorithm dominates all channels. National TV 
#      has rigid calendar seasonality (SARIMA favored), whereas Outdoor displays experience 
#      structural trend shifts (Meta Prophet favored).
#   3. Objective: Build an automated, self-selecting machine learning tournament that 
#      evaluates out-of-sample holdout accuracy (MAPE/RMSE) and produces 6-month forward forecasts.
# ==============================================================================

# ------------------------------------------------------------------------------
# 2. ENVIRONMENT SETUP & DEPENDENCY MANAGEMENT
# ------------------------------------------------------------------------------
# We load pacman to guarantee seamless library attachment
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}

pacman::p_load(
  readr,        # High-performance C++ CSV parser
  dplyr,        # Tidy grammar for relational filtering and aggregation
  tidyr,        # Data pivoting (pivot_longer / pivot_wider)
  ggplot2,      # Coherent grammar of graphics
  scales,       # Scientific, percentage, and currency formatting for plot axes
  lubridate,    # Arithmetic date operations (%m+%, year, month)
  psych,        # Classical psychological/econometric statistical profiling
  forecast,     # Hyndman's Box-Jenkins Auto-ARIMA & exponential smoothing suite
  prophet,      # Meta Prophet additive generalized forecasting framework
  writexl,      # Clean multi-tab Excel spreadsheet generation
  zoo,          # Year-month continuous temporal index handling
  shiny,        # Reactive web application UI/Server framework
  knitr         # Formatted Markdown/HTML tables
)

# ------------------------------------------------------------------------------
# 3. GLOBAL CONFIGURATION & HYPERPARAMETERS
# ------------------------------------------------------------------------------
CONFIG <- list(
  # File paths
  DATA_PATH         = "ASOMEDIO2012A2025MES.csv",
  EXCEL_OUTPUT      = "consolidated_media_forecasts.xlsx",
  
  # Time Series Modeling Parameters:
  #   - 'FREQUENCY' = 12: Monthly periodicity (s = 12). Necessary for SARIMA seasonal lag polynomials.
  #   - 'HOLDOUT_MONTHS' = 6: Out-of-sample test window. Models are trained on T - 6 months 
  #     and validated on the most recent 6 actuals to compute un-biased test MAPE/RMSE.
  #   - 'FORECAST_HORIZON' = 6: Length of the forward projection into unobserved future months.
  FREQUENCY         = 12,
  HOLDOUT_MONTHS    = 6,
  FORECAST_HORIZON  = 6,
  
  # Data Scaling:
  # Why divide by 100,000? Raw COP advertising spend spans billions. Dividing by 100,000
  # scales values into manageable index units, preventing numerical overflow and matrix ill-conditioning.
  SCALE_FACTOR      = 100000,
  
  # Evaluated Media Portfolio
  CHANNELS          = c("TV.REG.Y.LOCAL", "TV.NACIONAL", "RADIO", "REVISTAS", 
                        "PUB.EXTERIOR", "PERIODICOS", "TOTAL.CON.PERIODICOS"),
  
  THEME_BASE_SIZE   = 12
)

# ------------------------------------------------------------------------------
# 4. DATA INGESTION & TEMPORAL PREPARATION
# ------------------------------------------------------------------------------
if (!file.exists(CONFIG$DATA_PATH)) {
  stop(sprintf("Fatal Error: Time series dataset '%s' not found.", CONFIG$DATA_PATH))
}

# Ingest CSV with semicolon delimiter and comma decimal points
raw_medios <- read.csv(CONFIG$DATA_PATH, header = TRUE, sep = ";", dec = ",", stringsAsFactors = FALSE)

# Clean, sort chronologically, and parse date objects
df_medios <- raw_medios %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

# Normalize column names for safe programmatic access (replaces spaces with dots)
colnames(df_medios) <- make.names(colnames(df_medios))

# Rescale numeric advertising spend channels
numeric_cols <- intersect(CONFIG$CHANNELS, colnames(df_medios))
df_medios[numeric_cols] <- df_medios[numeric_cols] / CONFIG$SCALE_FACTOR

cat(sprintf("Loaded Asomedios time series from %s to %s (%d continuous monthly records).\n",
            min(df_medios$date), max(df_medios$date), nrow(df_medios)))

# ------------------------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS (EDA) & MULTI-SERIES DECOMPOSITION
# ------------------------------------------------------------------------------
cat("\n=== DESCRIPTIVE STATISTICAL PROFILE (ADVERTISING CHANNELS) ===\n")
# 'psych::describe' computes mean, median, trimmed mean, standard deviation, skewness, and kurtosis
desc_profile <- psych::describe(df_medios[numeric_cols])
print(round(desc_profile[, c("n", "mean", "sd", "median", "min", "max", "skew", "kurtosis")], 2))

# Multi-channel Historical Overview via Tidy Long Pivot
# Function Argument Breakdown:
#   - 'scales = "free_y"': Crucial! TV Nacional spends billions, while Magazines spend millions.
#     Decoupling the Y-axis allows accurate visual inspection of trend/seasonality across both large and small channels.
df_long <- df_medios %>%
  select(date, all_of(numeric_cols)) %>%
  pivot_longer(cols = -date, names_to = "Channel", values_to = "Spend")

p_overview <- ggplot(df_long, aes(x = date, y = Spend, color = Channel)) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~ Channel, scales = "free_y", ncol = 2) +
  scale_y_continuous(labels = comma) +
  labs(
    title = "Historical Monthly Advertising Investment by Channel (2012-2025)",
    subtitle = "Scaled Investment Units (Asomedios Dataset)",
    x = "Date",
    y = "Monthly Investment (Scaled Units)",
    caption = "Source: Asomedios Colombia"
  ) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none")

print(p_overview)

# ------------------------------------------------------------------------------
# 6. AUTOMATED FORECASTING TOURNAMENT ENGINE (SARIMA vs. META PROPHET)
# ------------------------------------------------------------------------------
#
# Model A: Seasonal Auto-Regressive Integrated Moving Average - SARIMA(p,d,q) x (P,D,Q)_s
# Mathematical Formulation (Hyndman-Khandakar, 2008):
#   Phi_P(B^s) * phi_p(B) * (1 - B)^d * (1 - B^s)^D * Y_t = Theta_Q(B^s) * theta_q(B) * epsilon_t
# where:
#   - B is the backshift lag operator: B^k * Y_t = Y_{t-k}.
#   - d is the order of regular non-seasonal differencing (ensuring mean stationarity).
#   - D is seasonal differencing: (1 - B^s)^D * Y_t (removing annual 12-month cycles).
#   - phi_p(B) and theta_q(B) are the non-seasonal AR and MA polynomials.
#   - Phi_P(B^s) and Theta_Q(B^s) are the seasonal AR and MA polynomials.
#
# Model B: Meta Prophet (Taylor & Letham, 2018)
# Mathematical Formulation (Generalized Additive Model):
#   y(t) = g(t) + s(t) + h(t) + epsilon_t
# where:
#   - g(t) is a piecewise linear/logistic trend with automatic changepoint selection:
#     g(t) = (k + a(t)^T delta) * t + (m + a(t)^T gamma).
#   - s(t) is periodic seasonality modeled via truncated Fourier series:
#     s(t) = sum_{n=1}^N [ a_n * cos(2*pi*n*t / P) + b_n * sin(2*pi*n*t / P) ].
#   - h(t) is holiday effects; epsilon_t is normally distributed error.
#
# Out-of-Sample Holdout Tournament Metrics:
#   - Root Mean Squared Error (RMSE): Penalizes large outlier forecast errors heavily.
#     RMSE = sqrt( 1/H * sum_{t=1}^H (Y_t - Y_hat_t)^2 )
#   - Mean Absolute Percentage Error (MAPE): Executive-friendly percentage error.
#     MAPE = (100% / H) * sum_{t=1}^H | (Y_t - Y_hat_t) / Y_t |

run_forecasting_tournament <- function(data_df, channel_name, config) {
  cat(sprintf("\n>>> RUNNING TOURNAMENT FOR CHANNEL: %s <<<\n", channel_name))
  
  series_values <- data_df[[channel_name]]
  valid_idx     <- !is.na(series_values)
  clean_dates   <- data_df$date[valid_idx]
  clean_values  <- series_values[valid_idx]
  
  start_year  <- year(min(clean_dates))
  start_month <- month(min(clean_dates))
  
  # Construct native R 'ts' object with monthly frequency (12)
  ts_full <- ts(clean_values, start = c(start_year, start_month), frequency = config$FREQUENCY)
  
  # Temporal Holdout Partitioning (Last 6 Months)
  # Why temporal holdout instead of random CV? Time series data has auto-correlation! 
  # Randomly sampling points causes severe data leakage (future leaking into past).
  n_total <- length(ts_full)
  n_train <- n_total - config$HOLDOUT_MONTHS
  
  ts_train <- subset(ts_full, end = n_train)
  ts_test  <- subset(ts_full, start = n_train + 1)
  test_actuals <- as.numeric(ts_test)
  
  # ----------------------------------------------------------------------------
  # 1. Fit SARIMA Model on Training Window
  # ----------------------------------------------------------------------------
  # Function Argument Breakdown:
  #   - 'seasonal = TRUE': Searches seasonal (P, D, Q)_12 space.
  #   - 'stepwise = TRUE': Uses greedy stepwise AICc search to find optimal order in O(log N) time.
  #   - 'approximation = FALSE': Calculates exact Gaussian likelihood rather than numerical shortcuts.
  fit_sarima <- auto.arima(ts_train, seasonal = TRUE, stepwise = TRUE, approximation = FALSE)
  fc_sarima  <- forecast(fit_sarima, h = config$HOLDOUT_MONTHS)
  pred_sarima<- as.numeric(fc_sarima$mean)
  
  rmse_sarima <- sqrt(mean((pred_sarima - test_actuals)^2))
  mape_sarima <- mean(abs((pred_sarima - test_actuals) / test_actuals)) * 100
  
  # ----------------------------------------------------------------------------
  # 2. Fit Meta Prophet Model on Training Window
  # ----------------------------------------------------------------------------
  # Prophet requires specific column names: 'ds' (Datestamp) and 'y' (Target value).
  df_prophet_train <- data.frame(
    ds = clean_dates[1:n_train],
    y  = clean_values[1:n_train]
  )
  
  # Function Argument Breakdown:
  #   - 'yearly.seasonality = TRUE': Fits annual Fourier series (P = 365.25 days).
  #   - 'weekly.seasonality = FALSE' & 'daily.seasonality = FALSE': Monthly aggregated data 
  #     does not contain intra-week or intra-day dynamics; disabling them avoids overfitting.
  fit_prophet <- suppressMessages(prophet(
    df_prophet_train, 
    yearly.seasonality = TRUE, 
    weekly.seasonality = FALSE, 
    daily.seasonality  = FALSE
  ))
  
  future_df   <- make_future_dataframe(fit_prophet, periods = config$HOLDOUT_MONTHS, freq = "month")
  fc_prophet  <- predict(fit_prophet, future_df)
  pred_prophet <- tail(fc_prophet$yhat, config$HOLDOUT_MONTHS)
  
  rmse_prophet <- sqrt(mean((pred_prophet - test_actuals)^2))
  mape_prophet <- mean(abs((pred_prophet - test_actuals) / test_actuals)) * 100
  
  # ----------------------------------------------------------------------------
  # 3. Model Selection Decision: The Tournament Winner
  # ----------------------------------------------------------------------------
  # We select the model that achieved the lowest Mean Absolute Percentage Error (MAPE) on unseen data.
  winner <- ifelse(mape_sarima <= mape_prophet, "SARIMA", "Prophet")
  
  cat(sprintf("SARIMA  -> RMSE: %.2f | Holdout MAPE: %.2f%%\n", rmse_sarima, mape_sarima))
  cat(sprintf("Prophet -> RMSE: %.2f | Holdout MAPE: %.2f%%\n", rmse_prophet, mape_prophet))
  cat(sprintf("🏆 TOURNAMENT WINNER: %s\n", winner))
  
  # ----------------------------------------------------------------------------
  # 4. Refit Winner Model on 100% of Available Data & Project 6 Months Ahead
  # ----------------------------------------------------------------------------
  # Lubridate date increment with '%m+%' safely adds calendar months without day overflow errors
  future_dates <- seq.Date(from = max(clean_dates) %m+% months(1), by = "month", length.out = config$FORECAST_HORIZON)
  
  if (winner == "SARIMA") {
    final_model <- auto.arima(ts_full, seasonal = TRUE)
    final_fc    <- forecast(final_model, h = config$FORECAST_HORIZON)
    
    forecast_table <- data.frame(
      Channel   = channel_name,
      Date      = future_dates,
      Model     = "SARIMA",
      Forecast  = as.numeric(final_fc$mean),
      Lower80   = as.numeric(final_fc$lower[, 1]),
      Upper80   = as.numeric(final_fc$upper[, 1]),
      Lower95   = as.numeric(final_fc$lower[, 2]),
      Upper95   = as.numeric(final_fc$upper[, 2])
    )
  } else {
    df_prophet_full <- data.frame(ds = clean_dates, y = clean_values)
    final_model     <- suppressMessages(prophet(
      df_prophet_full, 
      yearly.seasonality = TRUE, 
      weekly.seasonality = FALSE, 
      daily.seasonality  = FALSE
    ))
    future_full     <- make_future_dataframe(final_model, periods = config$FORECAST_HORIZON, freq = "month")
    fc_out          <- predict(final_model, future_full)
    future_preds    <- tail(fc_out, config$FORECAST_HORIZON)
    
    forecast_table <- data.frame(
      Channel   = channel_name,
      Date      = future_dates,
      Model     = "Prophet",
      Forecast  = future_preds$yhat,
      Lower80   = future_preds$yhat_lower,
      Upper80   = future_preds$yhat_upper,
      Lower95   = future_preds$yhat_lower,
      Upper95   = future_preds$yhat_upper
    )
  }
  
  # ----------------------------------------------------------------------------
  # 5. Executive Visualization with Uncertainty Ribbons (80% & 95% Confidence)
  # ----------------------------------------------------------------------------
  # Why show two confidence bands?
  # - 80% interval (darker blue): Expected operational budget range.
  # - 95% interval (lighter blue): Conservative stress-test / worst-case cash flow boundary.
  p_forecast <- ggplot(forecast_table, aes(x = Date, y = Forecast)) +
    geom_ribbon(aes(ymin = Lower95, ymax = Upper95), fill = "#9ecae1", alpha = 0.4) +
    geom_ribbon(aes(ymin = Lower80, ymax = Upper80), fill = "#3182bd", alpha = 0.4) +
    geom_line(color = "#08519c", linewidth = 1.2) +
    geom_point(color = "#08519c", size = 2) +
    scale_y_continuous(labels = comma) +
    labs(
      title = sprintf("6-Month Forecast: %s (Tournament Winner: %s)", channel_name, winner),
      subtitle = sprintf("Holdout Out-of-Sample Performance -> MAPE: %.2f%% | RMSE: %.2f", 
                         min(mape_sarima, mape_prophet), min(rmse_sarima, rmse_prophet)),
      x = "Future Date",
      y = "Projected Spend (Scaled Units)",
      caption = "Shaded ribbons indicate 80% (dark) and 95% (light) prediction intervals"
    ) +
    theme_minimal(base_size = config$THEME_BASE_SIZE)
  
  return(list(
    Channel        = channel_name,
    Winner         = winner,
    Metrics        = data.frame(SARIMA_MAPE = mape_sarima, Prophet_MAPE = mape_prophet, Winner = winner),
    Forecast_Table = forecast_table,
    Plot           = p_forecast
  ))
}

# ------------------------------------------------------------------------------
# 7. EXECUTE TOURNAMENT ACROSS ALL MEDIA CHANNELS (VECTORIZED)
# ------------------------------------------------------------------------------
# Iterates through all channels systematically without copy-pasted code blocks
tournament_results <- lapply(numeric_cols, function(col) {
  run_forecasting_tournament(df_medios, col, CONFIG)
})

# Display all generated forecast charts
for (res in tournament_results) {
  print(res$Plot)
}

# Consolidate all future projections into an enterprise data table
consolidated_forecasts <- bind_rows(lapply(tournament_results, function(r) r$Forecast_Table))

cat("\n=== CONSOLIDATED 6-MONTH FORWARD PROJECTIONS ===\n")
print(head(consolidated_forecasts, 12))

# Export to Excel for finance and budget committee review
write_xlsx(consolidated_forecasts, path = CONFIG$EXCEL_OUTPUT)
cat(sprintf("\nSuccessfully exported all channel forecasts to '%s'.\n", CONFIG$EXCEL_OUTPUT))

# ------------------------------------------------------------------------------
# 8. STRATEGIC YEAR-OVER-YEAR (YoY) BUDGET VARIANCE AUDIT
# ------------------------------------------------------------------------------
# Why do we do this?
# In financial planning and analysis (FP&A), executives assess budget growth:
#   YoY % Change = [ (Spend_Current_YTD - Spend_Previous_YTD) / Spend_Previous_YTD ] * 100
compute_yoy_variance <- function(data_df, target_year, target_month, channels) {
  data_df <- data_df %>%
    mutate(Year = year(date), Month = month(date))
  
  current_period <- data_df %>%
    filter(Year == target_year, Month <= target_month) %>%
    summarise(across(all_of(channels), ~ sum(.x, na.rm = TRUE)))
  
  previous_period <- data_df %>%
    filter(Year == (target_year - 1), Month <= target_month) %>%
    summarise(across(all_of(channels), ~ sum(.x, na.rm = TRUE)))
  
  yoy_table <- tibble(
    Channel        = channels,
    Previous_Spend = as.numeric(previous_period[1, channels]),
    Current_Spend  = as.numeric(current_period[1, channels])
  ) %>%
    mutate(
      Absolute_Change = Current_Spend - Previous_Spend,
      Percentage_Change = (Absolute_Change / Previous_Spend) * 100,
      Channel_Label   = ifelse(Channel == "TOTAL.CON.PERIODICOS", "TOTAL MARKET", Channel)
    )
  
  return(yoy_table)
}

# Audit YoY variance for the most recent year in the database
latest_year  <- max(year(df_medios$date))
latest_month <- max(month(df_medios$date[year(df_medios$date) == latest_year]))
yoy_summary  <- compute_yoy_variance(df_medios, latest_year, latest_month, numeric_cols)

cat(sprintf("\n=== YoY BUDGET VARIATION SUMMARY (Jan-%d %d vs. Jan-%d %d) ===\n",
            latest_month, latest_year, latest_month, latest_year - 1))
print(yoy_summary)

# ------------------------------------------------------------------------------
# 9. INTERACTIVE SHINY EXECUTIVE DASHBOARD
# ------------------------------------------------------------------------------
# Allows non-technical business stakeholders to interactively filter years, 
# compare year-to-date spending, and visually identify expanding vs contracting media channels.
run_media_shiny_app <- function(data_df, channels) {
  data_df <- data_df %>%
    mutate(Year = year(date), Month = month(date))
  
  available_years <- sort(unique(data_df$Year), decreasing = TRUE)
  
  ui <- fluidPage(
    titlePanel("Asomedios: Interannual Media Budget Intelligence"),
    sidebarLayout(
      sidebarPanel(
        selectInput("sel_year", "Select Evaluation Year:", choices = available_years, selected = available_years[1]),
        sliderInput("sel_month", "Accumulated Month (1 to 12):", min = 1, max = 12, value = 6, step = 1),
        helpText("Calculates cumulative spending vs. the identical period in the preceding year.")
      ),
      mainPanel(
        h4("Year-over-Year Performance Summary"),
        tableOutput("yoy_table"),
        plotOutput("yoy_chart", height = "400px")
      )
    )
  )
  
  server <- function(input, output, session) {
    yoy_reactive <- reactive({
      compute_yoy_variance(data_df, as.integer(input$sel_year), as.integer(input$sel_month), channels)
    })
    
    output$yoy_table <- renderTable({
      df <- yoy_reactive()
      df %>%
        mutate(
          Previous_Spend = comma(round(Previous_Spend, 1)),
          Current_Spend  = comma(round(Current_Spend, 1)),
          Absolute_Change= comma(round(Absolute_Change, 1)),
          Percentage_Change = sprintf("%.2f%%", Percentage_Change)
        )
    })
    
    output$yoy_chart <- renderPlot({
      df <- yoy_reactive()
      ggplot(df, aes(x = reorder(Channel_Label, Percentage_Change), y = Percentage_Change, fill = Percentage_Change >= 0)) +
        geom_col(show.legend = FALSE) +
        geom_text(aes(label = sprintf("%.1f%%", Percentage_Change)),
                  vjust = ifelse(df$Percentage_Change >= 0, -0.5, 1.2), fontface = "bold") +
        scale_fill_manual(values = c("TRUE" = "#2b83ba", "FALSE" = "#d7191c")) +
        labs(
          title = sprintf("YoY Budget Variance Percentage (%s)", input$sel_year),
          x = "Advertising Medium",
          y = "% Variation vs Previous Year"
        ) +
        theme_minimal(base_size = 13)
    })
  }
  
  shinyApp(ui = ui, server = server)
}

# Note: To launch the interactive dashboard, run:
# run_media_shiny_app(df_medios, numeric_cols)
