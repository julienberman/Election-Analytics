# Load libraries.
## install via `install.packages("name")`
library(ggplot2)
library(maps)
library(tidyverse)
library(usmap)
library(plotly)
library(gridExtra)
library(sf)
library(ggrepel)
library(shiny)
library(leaflet)
library(stargazer)
library(blogdown)
library(car)
library(lubridate)
library(zoo)
library(modelsummary)
library(GGally)
library(readxl)
library(caret)
library(CVXR)
library(glmnet)
library(stats)
library(formula.tools)
library(nloptr)
library(boot)
library(MASS)

# Aggregate economic indicators
rate_aggregation <- function(data, indicator, t1, t2) {
  subset <- data %>%
    filter(month_index >= t1 & month_index <= t2) %>%
    group_by(election_cycle) %>% 
    summarize(
      year = mean(election_year),
      value_t1 = first(.data[[indicator]][month_index == t1]),
      value_t2 = first(.data[[indicator]][month_index == t2]),
      # Dynamic naming of the aggregated variable
      !!paste0(indicator, "_agg") := (value_t2 - value_t1) / value_t1 * 100
    )
  return(subset %>% dplyr::select(-value_t1, -value_t2, -election_cycle))
}
aggregate_indicators <- function(df_full, df_fundamentals, indicators, period_start, period_end, aggregation_method) {
  agg_results <- map(indicators, ~aggregation_method(df_fundamentals, .x, period_start, period_end))
  combined_agg <- reduce(agg_results, full_join, by = "year") %>% 
    mutate(across(ends_with("_agg"), ~as.numeric(scale(.))))
  df_full_merged <- df_full %>%
    left_join(combined_agg, by = "year") %>%
    return(df_full_merged)
}

# Function to split state-level prediction data
split_state <- function(df, y) {
  subset <- df %>% 
    dplyr::select(
      year,
      state,
      abbr,
      electors,
      pl,
      pl_lag1, 
      pl_lag2,
      pl_d_2pv,
      pl_d_2pv_lag1,
      pl_d_2pv_lag2,
      hsa_adjustment, 
      rsa_adjustment, 
      elasticity,
      elasticity_lag1,
      elasticity_lag2,
      all_of(starts_with("cpr_")),
     # all_of(paste0("poll_lean_", 7:36)),
      all_of(paste0("poll_pv_lean_", 7:36))
    )
  train_data <- subset %>% filter(year < y)
  test_data <- subset %>% filter(year == y)
  return(list(train = train_data, test = test_data))
}

# Function to split national-level prediction data
split_national <- function(df, y) {
  subset <- df %>% 
    dplyr::select(-c(all_of(paste0("poll_margin_nat_", 0:6)), all_of(paste0("poll_pv_nat_", 0:6)))) %>% 
    group_by(year) %>% 
    summarize(
      margin_nat = first(margin_nat),
      d_2pv_nat = first(d_2pv_nat),
      jobs_agg = first(jobs_agg), 
      pce_agg = first(pce_agg), 
      rdpi_agg = first(rdpi_agg), 
      cpi_agg = first(cpi_agg), 
      ics_agg = first(ics_agg), 
      sp500_agg = first(sp500_agg), 
      unemp_agg = first(unemp_agg),
      weighted_avg_approval = first(weighted_avg_approval),
      incumb = first(incumb),
      incumb_party = first(incumb_party),
      across(starts_with("poll_margin_nat_"), ~ mean(.x, na.rm = TRUE)),
      across(starts_with("poll_pv_nat_"), ~ mean(.x, na.rm = TRUE))
    )
  
  train_data <- subset %>% filter(year < y)
  test_data <- subset %>% filter(year == y)
  return(list(train = train_data, test = test_data))
}

train_elastic_net <- function(df, formula) {
  
  # Create matrix of MSE for every alpha, lambda, year left out during cross validation
  alpha_range <- seq(0, 1, length.out = 11)
  lambda_range <- seq(0, 10, length.out = 101)
  years <- unique(df %>% pull(year))
  mse_matrix <- expand_grid(year = years, alpha = alpha_range, lambda = lambda_range) %>% 
    # create column for MSE, set to NA
    mutate(mse = NA_real_)
  
  count <- 1
  # iterate over the years
  for (year in years) {
    
    # Create validation data by excluding all rows from the current year
    train_data <- df %>% filter(year != !!year)
    val_data <- df %>% filter(year == !!year)
    
    # Create training model matrix from formula and training data
    mf_train <- model.frame(formula, data = train_data, na.action = na.pass) # keep NA values
    X_train <- model.matrix(formula, mf_train)[, -1]  # Remove intercept column
    y_train <- model.response(mf_train)
    
    # Create validation model matrix from formula and validation data
    mf_val <- model.frame(formula, data = val_data, na.action = na.pass) # keep columns with NA values
    X_val <- model.matrix(formula, mf_val)[, -1]  # Remove intercept column
    y_val <- model.response(mf_val)
    
    # Handle missing predictors
    X_train[is.na(X_train)] <- 0  # Replace NA with 0
    X_val[is.na(X_val)] <- 0
    
    # Iterate over alpha and lambda values
    for (alpha in alpha_range) {
      for (lambda in lambda_range) {
        
        # train the model at the given alpha and lambda levels
        model <- glmnet(X_train, y_train, alpha = alpha, lambda = lambda)
        
        # compute out of sample MSE on validation data
        predictions_val <- predict(model, newx = X_val)
        mse_val <- mean((predictions_val - y_val)^2, na.rm = TRUE)
        
        # store in corresponding slot of MSE matrix
        mse_matrix$mse[mse_matrix$year == year & 
                         mse_matrix$alpha == alpha & 
                         mse_matrix$lambda == lambda] <- mse_val
        
        #print(paste("Results: Iteration #", count))
        #print(paste("MSE: ", mse_val, ". alpha: ", alpha, ". lambda: ", lambda))
        #count <- count + 1
      }
    }
  }
  
  # calculate the minimum average MSE for each alpha and lambda
  avg_mse <- mse_matrix %>%
    group_by(alpha, lambda) %>%
    summarize(avg_mse = mean(mse, na.rm = TRUE), .groups = "drop")
  
  best_params <- avg_mse %>%
    filter(avg_mse == min(avg_mse)) %>%
    slice(1)
  
  best_alpha <- best_params$alpha
  best_lambda <- best_params$lambda
  min_avg_mse <- best_params$avg_mse
  
  # create full model matrix
  mf <- model.frame(formula, data = df, na.action = na.pass) # keep NA values
  X <- model.matrix(formula, mf)[, -1]  # Remove intercept column
  y <- model.response(mf)
  
  # Handle missing predictors
  X[is.na(X)] <- 0  # Replace NA with 0
  
  # train on full data
  model <- glmnet(X, y, alpha = best_alpha, lambda = best_lambda)
  
  # Calculate adjusted R-squared
  y_pred <- predict(model, newx = X)
  n <- nrow(X)
  p <- sum(coef(model) != 0) - 1  # number of non-zero coefficients (excluding intercept)
  tss <- sum((y - mean(y))^2)
  rss <- sum((y - y_pred)^2)
  r_squared <- 1 - (rss / tss)
  adj_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
  
  mse_plot <- avg_mse %>%
    ggplot(aes(x = alpha, y = lambda, fill = avg_mse)) +
    geom_tile() +
    scale_fill_viridis_c(trans = "log10") +
    labs(title = "Average MSE for different alpha and lambda values",
         x = "Alpha", y = "Lambda", fill = "Average MSE") +
    theme_minimal() +
    geom_point(data = best_params, aes(x = alpha, y = lambda), 
               color = "red", size = 3, shape = 4)
  
  # return the model, optimal alpha and lamba, R^2 and other in-sample errors, and out-of-sample MSE 
  list(
    model = model,
    formula = formula,
    alpha = best_alpha,
    lambda = best_lambda,
    y_pred = as.vector(y_pred),
    y_actual = as.vector(y),
    n = n,
    p = p,
    tss = tss,
    rss = rss,
    r_squared = r_squared,
    adj_r_squared = adj_r_squared,
    out_of_sample_mse = min_avg_mse,
    in_sample_mse = rss / n,
    mse_plot = mse_plot
  )
}

train_elastic_net_fast <- function(data, formula, n_bootstrap = 1000, min_obs_per_fold = 3, seed = NULL) {
  # set random state
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # Create model matrix from formula and data
  mf <- model.frame(formula, data = data, na.action = na.pass)
  X <- model.matrix(formula, mf)[, -1]  # Remove intercept column
  y <- model.response(mf)
  
  X[is.na(X)] <- 0  # Replace NA with 0
  
  # Set up grid for alpha and lambda
  alpha_grid <- seq(0, 1, length.out = 11)
  lambda_grid <- seq(0, 5, length.out = 101)
  
  # Calculate the maximum number of folds based on sample size
  n_obs <- nrow(X)
  max_folds <- min(10, floor(n_obs / min_obs_per_fold))
  
  if (max_folds < 3) {
    warning("Sample size is too small for cross-validation. Using 2-fold CV.")
    max_folds <- 2
  }
  
  # Perform cross-validation to find optimal alpha and lambda
  cv_results <- lapply(alpha_grid, function(a) {
    tryCatch({
      cv.glmnet(X, y, alpha = a, lambda = lambda_grid, nfolds = max_folds)
    }, error = function(e) {
      warning(paste("Error in cv.glmnet for alpha =", a, ":", e$message))
      NULL
    })
  })
  
  # Remove NULL results (if any)
  cv_results <- cv_results[!sapply(cv_results, is.null)]
  
  if (length(cv_results) == 0) {
    stop("All cross-validation attempts failed. Please check your data and model specification.")
  }
  
  # Find the best alpha and lambda
  avg_mse <- do.call(rbind, lapply(seq_along(cv_results), function(i) {
    data.frame(
      alpha = alpha_grid[i],
      lambda = cv_results[[i]]$lambda,
      avg_mse = cv_results[[i]]$cvm
    )
  }))
  
  best_params <- avg_mse %>%
    group_by(alpha) %>%
    slice(which.min(avg_mse)) %>%
    ungroup() %>%
    slice(which.min(avg_mse))
  
  best_alpha <- best_params$alpha
  best_lambda <- best_params$lambda
  
  # Fit final model with optimal parameters
  model <- glmnet(X, y, alpha = best_alpha, lambda = best_lambda)
  
  # Calculate various metrics
  y_pred <- predict(model, newx = X)
  n <- nrow(X)
  p <- ncol(X)
  
  tss <- sum((y - mean(y))^2)
  rss <- sum((y - y_pred)^2)
  r_squared <- 1 - (rss / tss)
  adj_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
  colnames <- colnames(X)
  
  
  # Create MSE plot
  mse_plot <- avg_mse %>%
    ggplot(aes(x = alpha, y = lambda, fill = avg_mse)) +
    geom_tile() +
    scale_fill_viridis_c(trans = "log10") +
    labs(title = "Average MSE for different alpha and lambda values",
         x = "Alpha", y = "Lambda", fill = "Average MSE") +
    theme_minimal() +
    geom_point(data = best_params, aes(x = alpha, y = lambda), 
               color = "red", size = 3, shape = 4)
  
  # Perform bootstrapping
  boot_func <- function(data, indices) {
    X_boot <- X[indices, ]
    y_boot <- y[indices]
    model_boot <- tryCatch({
      glmnet(X_boot, y_boot, alpha = best_alpha, lambda = best_lambda)
    }, error = function(e) {
      warning("Error in bootstrap iteration: ", e$message)
      return(NULL)
    })
    
    if (is.null(model_boot)) {
      return(NULL)
    }
    
    coef_boot <- coef(model_boot)
    # Ensure consistent length by filling with NA
    full_coef <- rep(NA, length(coef(model)))
    full_coef[match(rownames(coef_boot), rownames(coef(model)))] <- as.vector(coef_boot)
    
    # Add prediction for the entire dataset
    pred_boot <- predict(model_boot, newx = X)
    
    list(coef = full_coef, pred = as.vector(pred_boot))
  }
  
  boot_results <- replicate(n_bootstrap, boot_func(data, sample(nrow(X), replace = TRUE)), simplify = FALSE)
  
  # Remove NULL results
  boot_results <- boot_results[!sapply(boot_results, is.null)]
  
  if (length(boot_results) == 0) {
    stop("All bootstrap iterations failed. Please check your data and model specification.")
  }
  
  # Calculate standard errors for coefficients, handling NAs
  coef_matrix <- sapply(boot_results, function(x) x$coef)
  se_coef <- apply(coef_matrix, 1, function(x) sd(x, na.rm = TRUE))
  
  # Calculate standard errors for predictions
  pred_matrix <- sapply(boot_results, function(x) x$pred)
  se_pred <- apply(pred_matrix, 1, sd, na.rm = TRUE)
  
  # Prepare results
  coefficients <- coef(model)
  results <- data.frame(
    variable = rownames(coefficients),
    coefficient = as.vector(coefficients),
    std_error = se_coef
  )
  
  # Return comprehensive list of results
  list(
    model = model,
    formula = formula,
    alpha = best_alpha,
    lambda = best_lambda,
    X = X,
    colnames = colnames,
    y_pred = as.vector(y_pred),
    y_actual = as.vector(y),
    se_pred = se_pred,
    n = n,
    p = p,
    tss = tss,
    rss = rss,
    r_squared = r_squared,
    adj_r_squared = adj_r_squared,
    out_of_sample_mse = min(avg_mse$avg_mse),
    in_sample_mse = rss / n,
    mse_plot = mse_plot,
    results = results
  )
}

# get model coefficients
get_coef <- function(model_info) {
  return(as.vector(coef(model_info$model))) 
}

make_prediction <- function(model_info, new_data) {
  # Handle missing or NaN values
  new_data[is.na(new_data) | is.nan(as.matrix(new_data))] <- 0
  
  # Create the model matrix using the stored formula
  X_new <- model.matrix(model_info$formula, data = new_data, na.action = na.pass)[, -1]
  
  # Predict using the glmnet model
  point_estimate <- predict(model_info$model, newx = X_new, s = model_info$lambda)
  
  X_new <- as.matrix(X_new)
  if (ncol(X_new) == 1) {
    X_new <-t(X_new)
  }
    
  # Calculate prediction standard error
  X_old <- as.matrix(model_info$X)
  beta <- as.vector(coef(model_info$model))[-1]  # Remove intercept
  mse <- model_info$in_sample_mse
  
  cov_matrix <- X_new %*% solve(t(X_old) %*% X_old + diag(model_info$lambda, ncol(X_old))) %*% t(X_new)
  std_error <- sqrt(mse + mse*diag(cov_matrix))
  
  return(list(
      point_estimate = as.vector(point_estimate),
      std_error = as.vector(std_error)
    ))
}

make_prediction_old <- function(model_info, new_data) {
  # Handle missing or NaN values
  new_data[is.na(new_data) | is.nan(as.matrix(new_data))] <- 0
  
  # Create the model matrix using the stored formula
  X_new <- model.matrix(model_info$formula, data = new_data, na.action = na.pass)[, -1]
  
  # Predict using the glmnet model
  return(predict(model_info$model, newx = X_new, s = model_info$lambda))
}

# Stack models to create an ensemble prediction
train_ensemble <- function(model_infos) {
  
  # extract in-sample predictions
  in_sample_predictions <- lapply(model_infos, function(model_info) model_info$y_pred)
  
  # combine predictions into a new dataset
  meta_features <- do.call(cbind, in_sample_predictions) %>% as.data.frame()
  colnames(meta_features) <- paste0("model_", seq_along(model_infos))
  
  # train the meta-model
  y <- model_infos[[1]]$y_actual
  meta_model <- lm(y ~ . - 1, data = meta_features) # train on all columns
  
  # extract and normalize coefficients
  coef <- coef(meta_model)
  normalized_coef <- coef / sum(coef)
  
  # calculate y_pred using normalized coefficients
  y_pred <- as.matrix(meta_features) %*% normalized_coef
  
  # calculate r_squared and adjusted r_squared
  n <- length(y)
  p <- length(coef(meta_model)) - 1  # number of predictors (excluding intercept)
  tss <- sum((y - mean(y))^2)
  rss <- sum((y - y_pred)^2)
  r_squared <- 1 - (rss / tss)
  adj_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
  
  # Return the meta-model and associated statistics
  list(
    model = meta_model,
    y_pred = as.vector(y_pred),
    y_actual = as.vector(y),
    weights = normalized_coef,
    n = n,
    p = p,
    tss = tss,
    rss = rss,
    r_squared = r_squared,
    adj_r_squared = adj_r_squared
  )
}

train_ensemble_v2 <- function(model_infos) {
  
  # extract in-sample predictions
  predictions <- lapply(model_infos, function(model_info) model_info$y_pred) 
  
  # extract actual values
  y <- model_infos[[1]]$y_actual
  
  # Number of models
  n_models <- length(predictions)
  
  # Objective function (Mean Squared Error)
  obj_fun <- function(x) {
    stacked_pred <- Reduce(`+`, mapply(`*`, predictions, x, SIMPLIFY = FALSE))
    mse <- mean((stacked_pred - y)^2)
    return(mse)
  }
  
  # Constraint function
  const_fun <- function(x) {
    return(c(sum(x) - 1))  # Sum of coefficients = 1
  }
  
  # Set up optimization problem
  opts <- list("algorithm" = "NLOPT_LN_COBYLA",
               "xtol_rel" = 1.0e-8,
               "maxeval" = 1000)
  
  result <- nloptr(
    x0 = rep(1/n_models, n_models),  # Initial guess: equal weights
    eval_f = obj_fun,
    lb = rep(0, n_models),  # Lower bounds (non-negative coefficients)
    ub = rep(1, n_models),  # Upper bounds
    eval_g_ineq = const_fun,
    opts = opts
  )
  
  # Return the optimal coefficients
  coefficients <- result$solution
  names(coefficients) <- paste0("model_", seq_along(coefficients))
  
  return(list(
    coefficients = coefficients,
    mse = result$objective
  ))
}


# Function to calculate overfit
calculate_overfit <- function(MSE_out, MSE_in, n, p) {
  overfit <- MSE_out - ((n + p + 1) / (n - p - 1)) * MSE_in
  return(overfit)
}

# predict a state's elasticity
predict_elasticity <- function(train_data, test_data) {
  # Fit the linear model using training data
  model <- lm(elasticity ~ elasticity_lag1 + elasticity_lag2, data = train_data)
  
  # Predict the elasticity values for test_data using dplyr
  test_data$elasticity = predict(model, newdata = test_data)
  
  return(test_data)
}

# function to make prediction for ensemble model
make_ensemble_prediction <- function(ensemble_model_info, new_data) {
  # Make predictions using each sub-model
  predictions <- lapply(ensemble_model_info$sub_models, 
                        function(model_info) make_prediction(model_info, new_data))
  
  # Combine predictions into a data frame
  meta_features <- do.call(cbind, predictions) %>% as.data.frame()
  colnames(meta_features) <- paste0("model_", seq_along(predictions))
  
  # Use the meta-model to make the final prediction
  final_prediction <- predict(ensemble_model_info$model, newdata = meta_features)
  
  return(final_prediction)
}

# Define the color palette
map_colors <- tibble(
  category = c("Strong R", "Likely R", "Lean R", "Lean D", "Likely D", "Strong D"),
  colors = c("#e48782", "#f0bbb8", "#fbeeed", "#e5f3fd", "#6ac5fe", "#0276ab")
) %>%
  deframe()

# Join the hex coordinates with our prediction data
create_electoral_hex_map <- function(prediction_data) {
  # Define hex coordinates using proper hexagonal grid spacing
  # In a hex grid, horizontal spacing is 1 unit, vertical is 0.866 units (sin(60°))
  hex_coords <- tibble(
    abbr = c(
      # Row 1 (top)
      "AK",                                                              "ME",
      # Row 2
      "WA",                                                    "VT", "NH", "ME_d2",
      # Row 3
      "OR", "MT", "ND", "MN", "WI",      "MI",              "NY", "MA", "RI",
      # Row 4
      "CA", "ID", "WY", "SD", "IA", "IL", "IN", "OH", "PA", "NJ", "CT",
      # Row 5
      "NV", "CO", "NE", "NE_d2", "MO", "KY", "WV", "VA", "MD", "DE",
      # Row 6
      "AZ", "UT", "NM", "KS", "AR", "TN", "NC", "SC", "DC",
      # Row 7
      "OK", "LA", "MS", "AL", "GA",
      # Row 8
      "HI",      "TX",                    "FL"
    ),
    x = c(
      # Row 1
      1,                                            11,
      # Row 2
      1,                                   9, 10, 11,
      # Row 3
      1, 2, 3, 4, 5,              7,     9, 10, 11,
      # Row 4
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
      # Row 5
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
      # Row 6
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      # Row 7
      3, 4, 5, 6, 7,
      # Row 8
      0,       3,                     8
    ),
    y = c(
      # Row 1
      9,                                           9,
      # Row 2
      8,                                    8, 8, 8,
      # Row 3
      7, 7, 7, 7, 7,    7,              7, 7, 7,
      # Row 4
      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      # Row 5
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      # Row 6
      4, 4, 4, 4, 4, 4, 4, 4, 4,
      # Row 7
      3, 3, 3, 3, 3,
      # Row 8
      2,       2,                     2
    )
  )
  
  # Calculate proper hexagonal spacing
  hex_size <- 40  # Base size of hexagons
  hex_width <- hex_size * 2
  hex_height <- hex_size * sqrt(3)
  
  # Apply hexagonal grid transformations
  hex_coords <- hex_coords %>%
    mutate(
      # Offset every other row horizontally by half a hex width
      x = case_when(
        y %% 2 == 0 ~ x * hex_width,
        TRUE ~ x * hex_width + hex_width/2
      ),
      # Scale y coordinates to create proper hexagonal spacing
      y = y * hex_height * 0.75  # 0.75 factor for tighter vertical spacing
    )
  
  map_data <- prediction_data %>%
    left_join(hex_coords, by = "abbr") %>%
    mutate(
      hover_text = sprintf(
        "%s\nVotes: %d\nDem: %.1f%%\nRep: %.1f%%",
        abbr, 
        electors,
        d_pv,
        r_pv
      )
    )
  
  hex_map <- plot_ly(
    data = map_data,
    type = "scatter",
    mode = "markers",
    x = ~x,
    y = ~y,
    color = ~category,
    colors = map_colors,
    marker = list(
      symbol = "hexagon",
      size = hex_size,
      line = list(color = "white", width = 1)
    ),
    text = ~hover_text,
    hoverinfo = "text"
  ) %>%
    layout(
      title = list(
        text = "2024 Electoral College Prediction",
        x = 0.5,
        y = 0.95
      ),
      showlegend = TRUE,
      xaxis = list(
        showgrid = FALSE,
        zeroline = FALSE,
        showticklabels = FALSE,
        range = c(-50, hex_width * 12),
        title = ""
      ),
      yaxis = list(
        showgrid = FALSE,
        zeroline = FALSE,
        showticklabels = FALSE,
        range = c(0, hex_height * 10),
        scaleanchor = "x",  # This ensures hexagons remain regular
        scaleratio = 1,
        title = ""
      ),
      plot_bgcolor = "white",
      paper_bgcolor = "white"
    )
  
  # Add state labels
  hex_map <- hex_map %>%
    add_annotations(
      data = map_data,
      x = ~x,
      y = ~y,
      text = ~abbr,
      showarrow = FALSE,
      font = list(size = 12, color = "black")
    )
  
  # Add electoral vote totals
  subtitle <- sprintf(
    "Democratic EVs: %d | Republican EVs: %d",
    unique(map_data$d_electors),
    unique(map_data$r_electors)
  )
  
  hex_map <- hex_map %>%
    layout(
      annotations = list(
        list(
          x = 0.5,
          y = -0.1,
          text = subtitle,
          showarrow = FALSE,
          xref = "paper",
          yref = "paper",
          font = list(size = 14)
        )
      )
    )
  
  return(hex_map)
}


simulate_election <- function(df_new, n_simulations = 100) {
  # initialize a matrix to store results
  results <- tibble(
    simulation = 1:n_simulations,
    d_ev = numeric(n_simulations),
    r_ev = numeric(n_simulations)
  )
  
  # extract national-level parameters
  d_2pv_polls <- df_2024$d_2pv_polls[1]
  d_2pv_polls_se <- df_2024$d_2pv_polls_se[1]
  
  for (i in 1:n_simulations) {
    
    # simulate national two-party vote share
    d_2pv_sim <- rnorm(1, mean = d_2pv_polls, sd = d_2pv_polls_se)
    
    # simulate state-level results and calculate electoral votes
    sim_results <- df_2024 %>%
      mutate(
        state_result = rnorm(n(), mean = partisan_lean + d_2pv_sim, sd = partisan_lean_se),
        dem_win = state_result > 50
      ) %>%
      summarise(
        d_ev = sum(electors * dem_win),
        r_ev = sum(electors * (!dem_win))
      )
    # store results
    results[i, c("d_ev", "r_ev")] <- sim_results
  }
  return(results)
  
}


# simulate election
simulate_election_old <- function(nat_polls_model_info, state_model_info, national_data_test, state_data_test, n_simulations = 10000) {
  
  # Function to simulate coefficients
  simulate_coefficients <- function(model_info) {
    coef_mean <- model_info$results$coefficient
    coef_se <- model_info$results$std_error
    mvrnorm(n = 1, mu = coef_mean, Sigma = diag(coef_se^2))
  }
  
  # Function to make predictions using simulated coefficients
  predict_with_simulated_coef <- function(model_info, new_data, simulated_coef) {
    # Handle missing or NaN values
    new_data[is.na(new_data) | is.nan(as.matrix(new_data))] <- 0
    
    # Create the model matrix using the stored formula
    X_new <- model.matrix(model_info$formula, data = new_data, na.action = na.pass)
    
    as.vector(X_new %*% simulated_coef)
  }
  
  # Initialize results storage
  simulation_results <- vector("list", n_simulations)
  
  for (i in 1:n_simulations) {
    # Simulate coefficients
    nat_polls_coef <- simulate_coefficients(nat_polls_model_info)
    state_coef <- simulate_coefficients(state_model_info)
    
    # Make predictions
    nat_prediction <- predict_with_simulated_coef(nat_polls_model_info, national_data_test, nat_polls_coef)
    state_predictions <- predict_with_simulated_coef(state_model_info, state_data_test, state_coef)
    
    # Create results dataframe
    state_data_test$pl_d_2pv <- state_predictions
    
    results <- state_data_test %>%
      mutate(
        d_2pv_nat = nat_prediction[1],  # Use only the first value of nat_prediction
        d_2pv_final = pl_d_2pv + d_2pv_nat,
        winner = ifelse(d_2pv_final > 50, "Democrat", "Republican"),
        d_ev = ifelse(winner == "Democrat", electors, 0),
        r_ev = ifelse(winner == "Republican", electors, 0)
      )
    
    
    simulation_results[[i]] <- list(d_ev = sum(results$d_ev), r_ev = sum(results$r_ev), d_2pv_nat = results$d_2pv_nat[1])
  }
  
  # Summarize all simulations
  summary <- tibble(
    d_ev = sapply(simulation_results, function(x) x$d_ev),
    r_ev = sapply(simulation_results, function(x) x$r_ev),
    d_2pv_nat = sapply(simulation_results, function(x) x$d_2pv_nat)
  )
  
  return(summary)
}


