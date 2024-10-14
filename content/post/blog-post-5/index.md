---
title: 'Blog Post 5: Quantifying Uncertainty'
author: Package Build
date: '2024-10-06'
slug: blog-post-5
categories: []
tags: []
---

<link href="{{< blogdown/postref >}}index_files/htmltools-fill/fill.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/htmlwidgets/htmlwidgets.js"></script>
<script src="{{< blogdown/postref >}}index_files/plotly-binding/plotly.js"></script>
<script src="{{< blogdown/postref >}}index_files/typedarray/typedarray.min.js"></script>
<script src="{{< blogdown/postref >}}index_files/jquery/jquery.min.js"></script>
<link href="{{< blogdown/postref >}}index_files/crosstalk/css/crosstalk.min.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/crosstalk/js/crosstalk.min.js"></script>
<link href="{{< blogdown/postref >}}index_files/plotly-htmlwidgets-css/plotly-htmlwidgets.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index_files/plotly-main/plotly-latest.min.js"></script>

Last week, I finally completed the full extent of my model architecture! We have now incorporated most of the data that we will use in our final model, multiple layers of prediction using an elastic net regression with optimized parameters, and model ensembling. Three items are on our agenda for this week. First, I will fix a few lingering errors that I noticed in last week’s model. Then, I will build a visualization for my 2024 election predictions. And finally, I will attempt to implement a Monte Carlo simulation that tests the uncertainty of my model.

# Error Correction

I noticed two errors in my previous model. First, last week, I incorrectly assumed that the ensemble learning weights do not have to sum to one. Turns out, I misinterpreted super learning as an unconstrained optimization problem. In fact, the idea is to take a convex combination of each base model and produce a single prediction, which means that the weights actually do have to [add to one](https://www.degruyter.com/document/doi/10.2202/1544-6115.1309/html?lang=en). In this week’s model, I add that constraint.

Second, I may need to modify my method for imputing missing data. Currently, there are many indicators — both polls and economic indicators — that do not extend as far back in time as my output variable. However, in previous models, I thought it would be remiss to forego a few extra years of training simply because a few indicators are missing. In fact, there is so much data missing, that if I threw out every state or every year in which an indicator was missing, I would end up with hardly any training data at all! Because there is so much missing data, it is also not impossible to impute. After all, I can’t just invent polling data for an entire year!

So, what I have done in previous weeks is to simply impute the missing indicators as zero. That way, when I train the model, the coefficients on the missing indicators will not effect the coefficients of the non-missing indicators. However, this approach, while plausible, could potentially bias the coefficients of both indicators themselves. (Here, when I say “missing” indicators, I am referring indicators for which data during some subset of the years from 1952 to 2022 are missing.) For example, suppose poll_margin_nat_7 — the indicator for the national polling margin seven weeks before the election — is missing from the years 1952 to 2000, and accurate from 2000 to 2020. Then, the coefficient of the polling variable is likely biased downward because, for the earlier period, I have assumed that polling had no effect on the vote margin (when in reality, it likely did). Similarly, the other variables are likely biased upwards, because they could be picking up some of the variation that should have been explained by polling.

Unfortunately, this issue isn’t easy to solve. I can minimize the bias by excluding years with lots of missing indicators from my dataset, but that reduces that already-small sample I have to train on, which could cause overfitting and make both my in-sample and out-of-sample predictions less accurate. To be rigorous about this, let’s define a precise “overfitting” metric as the differential between the out-of-sample mean squared error of a given model and some function of the in-sample mean squared error of that model.

If the model is correctly specified, it can be shown under mild assumptions that the expected value of the MSE for the training set (i.e. our in-sample MSE) is (n − p − 1)/(n + p + 1) \< 1 times the expected value of the MSE for the validation set (i.e. our out-of-sample MSE), where n is the number of observations, and p is the number of features. Luckily,it is possible to directly compute the factor (n − p − 1)/(n + p + 1) by which the training MSE underestimates the validation MSE. So, we can create our “overfitted” metric as:
$$
\mathrm{overfit} = \mathrm{MSE\_out} - \left(\frac{n + p + 1}{n − p − 1}\right) \mathrm{MSE\_in}
$$

The following table reports the overfitting metric for each of three potential subsets of the data:

``` r
# Define formulas
state_formula <- as.formula(paste("pl ~ pl_lag1 + pl_lag2 + hsa_adjustment +",
                                  "rsa_adjustment + elasticity +", 
                                  "cpr_solid_d + cpr_likely_d	+ cpr_lean_d +", 
                                  "cpr_toss_up + cpr_lean_r + cpr_likely_r	+ cpr_solid_r + ",
                                  paste0("poll_lean_", 7:36, collapse = " + ")))

nat_fund_formula <- as.formula("margin_nat ~ incumb_party:(jobs_agg + 
                                     pce_agg + rdpi_agg + cpi_agg + ics_agg + 
                                     sp500_agg + unemp_agg)")

nat_polls_formula <- as.formula(paste("margin_nat ~ incumb_party:(weighted_avg_approval) + ", 
                                           paste0("poll_margin_nat_", 7:36, collapse = " + ")))

# create lists of dataframes for comparison
df_subset_1972 <- df %>% filter(year >= 1972)
df_subset_1980 <- df %>% filter(year >= 1980)
df_subset_2000 <- df %>% filter(year >= 2000) 
dfs <-  list(df, df_subset_1972, df_subset_1980, df_subset_2000)

# Initialize a matrix to store the MSEs (3 models x 4 subsets)
mse_matrix <- matrix(nrow = 4, ncol = 3)
rownames(mse_matrix) <- c("Full Data", "Subset >= 1972", "Subset >= 1980", "Subset >= 2000")
colnames(mse_matrix) <- c("State Model", "Nat Fund Model", "Nat Polls Model")

for (i in seq_along(dfs)) {
  # access df
  df <- dfs[[i]]
  
  # Split data
  state_data <- split_state(df, 2024)
  national_data <- split_national(df, 2024)
  
  # Train models
  state_model_info <- train_elastic_net(state_data$train, state_formula)
  nat_fund_model_info <- train_elastic_net(national_data$train, nat_fund_formula)
  nat_polls_model_info <- train_elastic_net(national_data$train, nat_polls_formula)

  # Store MSEs in the matrix
  mse_matrix[i, 1] <- calculate_overfit(state_model_info$out_of_sample_mse, state_model_info$in_sample_mse,
                                        state_model_info$n, state_model_info$p)
  mse_matrix[i, 2] <- calculate_overfit(nat_fund_model_info$out_of_sample_mse, nat_fund_model_info$in_sample_mse,
                                        nat_fund_model_info$n, nat_fund_model_info$p)
  mse_matrix[i, 3] <- calculate_overfit(nat_polls_model_info$out_of_sample_mse, nat_polls_model_info$in_sample_mse,
                                        nat_polls_model_info$n, nat_polls_model_info$p)
}

print(mse_matrix)
```

    ##                State Model Nat Fund Model Nat Polls Model
    ## Full Data         6.403523       7.848683    -145.1184452
    ## Subset >= 1972    4.333628     -21.224004    -248.5398521
    ## Subset >= 1980    6.019361      -9.276520      21.6290282
    ## Subset >= 2000   10.246393      -4.000452       0.3900102

This testing suggests that the 1972 subset is best, which is what I will use for the remainder of the blog.

# Visualizing the prediction

Voila! Below, find my predicted electoral map for the 2024 election. I used an interactive hex map, visualized with plotly, using the standard coordinates for the US states and districts. There are still some issues with this prediction. For one, the prediction for Washington D.C. says that the Democrats will win approximately 104 percent of the vote share. Now, I’m from D.C. — and trust me: we vote very blue. But I know for a fact that there’s no way the Democrats win 104 percent of our residents! Something is clearly wrong, likely because my outcome variable is not bounded. This will be fixed in future blog iterations.

``` r
# Define formulas
state_formula <- as.formula(paste("pl ~ pl_lag1 + pl_lag2 + hsa_adjustment +",
                                  "rsa_adjustment + elasticity +", 
                                  "cpr_solid_d + cpr_likely_d	+ cpr_lean_d +", 
                                  "cpr_toss_up + cpr_lean_r + cpr_likely_r	+ cpr_solid_r + ",
                                  paste0("poll_lean_", 7:36, collapse = " + ")))

nat_fund_formula <- as.formula("margin_nat ~ incumb_party:(jobs_agg + 
                                     pce_agg + rdpi_agg + cpi_agg + ics_agg + 
                                     sp500_agg + unemp_agg)")

nat_polls_formula <- as.formula(paste("margin_nat ~ incumb_party:(weighted_avg_approval) + ", 
                                           paste0("poll_margin_nat_", 7:36, collapse = " + ")))

# Split data, using the 1972 subset
state_data <- split_state(df_subset_1972, 2024)
national_data <- split_national(df_subset_1972, 2024)

# Train models
state_model_info <- train_elastic_net(state_data$train, state_formula)
nat_fund_model_info <- train_elastic_net(national_data$train, nat_fund_formula)
nat_polls_model_info <- train_elastic_net(national_data$train, nat_polls_formula)
ensemble <- train_ensemble(list(nat_fund_model_info, nat_polls_model_info))

# Make predictions
state_predictions <- make_prediction(state_model_info, state_data$test)
nat_fund_predictions <- make_prediction(nat_fund_model_info, national_data$test)
nat_polls_predictions <- make_prediction(nat_polls_model_info, national_data$test)
ensemble_predictions <- make_ensemble_prediction(ensemble, national_data$test)

# Create the prediction tibble
df_2024 <- tibble(
  state = state_data$test$state,
  abbr = state_data$test$abbr,
  electors = state_data$test$electors,
  partisan_lean = as.vector(state_predictions)
  ) %>%
  # filter unnecessary districts
  filter(!abbr %in% c("ME_d1", "NE_d1", "NE_d3")) %>% 
  # Add national predictions - using first value since they're the same for all states
  mutate(
    margin_polls = first(as.vector(nat_polls_predictions)),
    margin_fund = first(as.vector(nat_fund_predictions)),
    margin_ensemble = first(as.vector(ensemble_predictions))
  ) %>%
  # Calculate final margins and color categories
  mutate(
    margin_final = partisan_lean + margin_ensemble,
    d_pv = margin_final + 50,
    r_pv = 100 - d_pv,
    category = case_when(
      d_pv > 60 ~ "Strong D",
      d_pv > 55 & d_pv < 60 ~ "Likely D",
      d_pv > 50 & d_pv < 55 ~ "Lean D",
      d_pv > 45 & d_pv < 50 ~ "Lean R",
      d_pv > 40 & d_pv < 45 ~ "Likely R",
      TRUE ~ "Strong R"
    ),
    # Convert color_category to factor with specific ordering
    category = factor(
      category,
      levels = c("Strong R", "Likely R", "Lean R", "Lean D", "Likely D", "Strong D")
    ),
    # calculate electors that each party wins
    d_electors = sum(ifelse(category %in% c("Lean D", "Likely D", "Strong D"), electors, 0)),
    r_electors = sum(ifelse(category %in% c("Lean R", "Likely R", "Strong R"), electors, 0))
  )

electoral_map <- create_electoral_hex_map(df_2024)
electoral_map
```

<div class="plotly html-widget html-fill-item" id="htmlwidget-1" style="width:672px;height:480px;"></div>
<script type="application/json" data-for="htmlwidget-1">{"x":{"visdat":{"638f12c2045a":["function () ","plotlyVisDat"],"638f7fabba54":["function () ","data"]},"cur_data":"638f7fabba54","attrs":{"638f12c2045a":{"mode":"markers","x":{},"y":{},"marker":{"symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":{},"hoverinfo":"text","color":{},"colors":["#e48782","#f0bbb8","#fbeeed","#e5f3fd","#6ac5fe","#0276ab"],"alpha_stroke":1,"sizes":[10,100],"spans":[1,20],"type":"scatter"}},"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"title":{"text":"2024 Electoral College Prediction","x":0.5,"y":0.94999999999999996},"showlegend":true,"xaxis":{"domain":[0,1],"automargin":true,"showgrid":false,"zeroline":false,"showticklabels":false,"range":[-50,960],"title":""},"yaxis":{"domain":[0,1],"automargin":true,"showgrid":false,"zeroline":false,"showticklabels":false,"range":[0,692.82032302755078],"scaleanchor":"x","scaleratio":1,"title":""},"plot_bgcolor":"white","paper_bgcolor":"white","annotations":[{"text":"AL","x":520,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"AK","x":120,"y":467.65371804359677,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"AZ","x":80,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"AR","x":400,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"CA","x":80,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"CO","x":200,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"CT","x":880,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"DE","x":840,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"DC","x":720,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"FL","x":640,"y":103.92304845413263,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"GA","x":600,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"HI","x":0,"y":103.92304845413263,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"ID","x":160,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"IL","x":480,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"IN","x":560,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"IA","x":400,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"KS","x":320,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"KY","x":520,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"LA","x":360,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"ME","x":920,"y":467.65371804359677,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"ME_d2","x":880,"y":415.69219381653051,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MD","x":760,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MA","x":840,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MI","x":600,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MN","x":360,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MS","x":440,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MO","x":440,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MT","x":200,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NE","x":280,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NE_d2","x":360,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NV","x":120,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NH","x":800,"y":415.69219381653051,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NJ","x":800,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NM","x":240,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NY","x":760,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NC","x":560,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"ND","x":280,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"OH","x":640,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"OK","x":280,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"OR","x":120,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"PA","x":720,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"RI","x":920,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"SC","x":640,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"SD","x":320,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"TN","x":480,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"TX","x":240,"y":103.92304845413263,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"UT","x":160,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"VT","x":720,"y":415.69219381653051,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"VA","x":680,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"WA","x":80,"y":415.69219381653051,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"WV","x":600,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"WI","x":440,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"WY","x":240,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"x":0.5,"y":-0.10000000000000001,"text":"Democratic EVs: 226 | Republican EVs: 309","showarrow":false,"xref":"paper","yref":"paper","font":{"size":14}}],"hovermode":"closest"},"source":"A","config":{"modeBarButtonsToAdd":["hoverclosest","hovercompare"],"showSendToCloud":false},"data":[{"mode":"markers","x":[520,120,400,160,560,400,320,520,360,440,440,200,280,280,640,280,640,320,480,160,600,240],"y":[155.88457268119893,467.65371804359677,207.84609690826525,311.76914536239786,311.76914536239786,311.76914536239786,207.84609690826525,259.80762113533154,155.88457268119893,155.88457268119893,259.80762113533154,363.73066958946424,259.80762113533154,363.73066958946424,311.76914536239786,155.88457268119893,207.84609690826525,311.76914536239786,207.84609690826525,207.84609690826525,259.80762113533154,311.76914536239786],"marker":{"color":"rgba(228,135,130,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["AL<br />Votes: 9<br />Dem: 27.9%<br />Rep: 72.1%","AK<br />Votes: 3<br />Dem: 37.0%<br />Rep: 63.0%","AR<br />Votes: 6<br />Dem: 25.2%<br />Rep: 74.8%","ID<br />Votes: 4<br />Dem: 22.7%<br />Rep: 77.3%","IN<br />Votes: 11<br />Dem: 34.3%<br />Rep: 65.7%","IA<br />Votes: 6<br />Dem: 39.7%<br />Rep: 60.3%","KS<br />Votes: 6<br />Dem: 35.2%<br />Rep: 64.8%","KY<br />Votes: 8<br />Dem: 27.3%<br />Rep: 72.7%","LA<br />Votes: 8<br />Dem: 32.7%<br />Rep: 67.3%","MS<br />Votes: 6<br />Dem: 34.4%<br />Rep: 65.6%","MO<br />Votes: 10<br />Dem: 35.0%<br />Rep: 65.0%","MT<br />Votes: 4<br />Dem: 33.7%<br />Rep: 66.3%","NE<br />Votes: 2<br />Dem: 31.7%<br />Rep: 68.3%","ND<br />Votes: 3<br />Dem: 20.8%<br />Rep: 79.2%","OH<br />Votes: 17<br />Dem: 38.3%<br />Rep: 61.7%","OK<br />Votes: 7<br />Dem: 21.6%<br />Rep: 78.4%","SC<br />Votes: 9<br />Dem: 37.9%<br />Rep: 62.1%","SD<br />Votes: 3<br />Dem: 26.5%<br />Rep: 73.5%","TN<br />Votes: 11<br />Dem: 29.2%<br />Rep: 70.8%","UT<br />Votes: 6<br />Dem: 29.4%<br />Rep: 70.6%","WV<br />Votes: 4<br />Dem: 17.8%<br />Rep: 82.2%","WY<br />Votes: 3<br />Dem: 12.9%<br />Rep: 87.1%"],"hoverinfo":["text","text","text","text","text","text","text","text","text","text","text","text","text","text","text","text","text","text","text","text","text","text"],"type":"scatter","name":"Strong R","textfont":{"color":"rgba(228,135,130,1)"},"error_y":{"color":"rgba(228,135,130,1)"},"error_x":{"color":"rgba(228,135,130,1)"},"line":{"color":"rgba(228,135,130,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[640,560,240],"y":[103.92304845413263,207.84609690826525,103.92304845413263],"marker":{"color":"rgba(240,187,184,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["FL<br />Votes: 30<br />Dem: 42.9%<br />Rep: 57.1%","NC<br />Votes: 16<br />Dem: 44.7%<br />Rep: 55.3%","TX<br />Votes: 40<br />Dem: 40.5%<br />Rep: 59.5%"],"hoverinfo":["text","text","text"],"type":"scatter","name":"Likely R","textfont":{"color":"rgba(240,187,184,1)"},"error_y":{"color":"rgba(240,187,184,1)"},"error_x":{"color":"rgba(240,187,184,1)"},"line":{"color":"rgba(240,187,184,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[80,600,600,120,720,440],"y":[207.84609690826525,155.88457268119893,363.73066958946424,259.80762113533154,311.76914536239786,363.73066958946424],"marker":{"color":"rgba(251,238,237,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["AZ<br />Votes: 11<br />Dem: 45.9%<br />Rep: 54.1%","GA<br />Votes: 16<br />Dem: 46.0%<br />Rep: 54.0%","MI<br />Votes: 15<br />Dem: 48.3%<br />Rep: 51.7%","NV<br />Votes: 6<br />Dem: 47.1%<br />Rep: 52.9%","PA<br />Votes: 19<br />Dem: 47.2%<br />Rep: 52.8%","WI<br />Votes: 10<br />Dem: 46.8%<br />Rep: 53.2%"],"hoverinfo":["text","text","text","text","text","text"],"type":"scatter","name":"Lean R","textfont":{"color":"rgba(251,238,237,1)"},"error_y":{"color":"rgba(251,238,237,1)"},"error_x":{"color":"rgba(251,238,237,1)"},"line":{"color":"rgba(251,238,237,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[200,880,360,360,800,240,680],"y":[259.80762113533154,415.69219381653051,363.73066958946424,259.80762113533154,415.69219381653051,207.84609690826525,259.80762113533154],"marker":{"color":"rgba(229,243,253,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["CO<br />Votes: 10<br />Dem: 54.9%<br />Rep: 45.1%","ME_d2<br />Votes: 1<br />Dem: 51.3%<br />Rep: 48.7%","MN<br />Votes: 10<br />Dem: 51.8%<br />Rep: 48.2%","NE_d2<br />Votes: 1<br />Dem: 50.5%<br />Rep: 49.5%","NH<br />Votes: 4<br />Dem: 52.3%<br />Rep: 47.7%","NM<br />Votes: 5<br />Dem: 53.8%<br />Rep: 46.2%","VA<br />Votes: 13<br />Dem: 54.6%<br />Rep: 45.4%"],"hoverinfo":["text","text","text","text","text","text","text"],"type":"scatter","name":"Lean D","textfont":{"color":"rgba(229,243,253,1)"},"error_y":{"color":"rgba(229,243,253,1)"},"error_x":{"color":"rgba(229,243,253,1)"},"line":{"color":"rgba(229,243,253,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[480,800,120,80],"y":[311.76914536239786,311.76914536239786,363.73066958946424,415.69219381653051],"marker":{"color":"rgba(106,197,254,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["IL<br />Votes: 19<br />Dem: 58.6%<br />Rep: 41.4%","NJ<br />Votes: 14<br />Dem: 58.3%<br />Rep: 41.7%","OR<br />Votes: 8<br />Dem: 57.0%<br />Rep: 43.0%","WA<br />Votes: 12<br />Dem: 59.2%<br />Rep: 40.8%"],"hoverinfo":["text","text","text","text"],"type":"scatter","name":"Likely D","textfont":{"color":"rgba(106,197,254,1)"},"error_y":{"color":"rgba(106,197,254,1)"},"error_x":{"color":"rgba(106,197,254,1)"},"line":{"color":"rgba(106,197,254,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[80,880,840,720,0,920,760,840,760,920,720],"y":[311.76914536239786,311.76914536239786,259.80762113533154,207.84609690826525,103.92304845413263,467.65371804359677,259.80762113533154,363.73066958946424,363.73066958946424,363.73066958946424,415.69219381653051],"marker":{"color":"rgba(2,118,171,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["CA<br />Votes: 54<br />Dem: 71.3%<br />Rep: 28.7%","CT<br />Votes: 7<br />Dem: 61.1%<br />Rep: 38.9%","DE<br />Votes: 3<br />Dem: 60.4%<br />Rep: 39.6%","DC<br />Votes: 3<br />Dem: 109.3%<br />Rep: -9.3%","HI<br />Votes: 4<br />Dem: 66.8%<br />Rep: 33.2%","ME<br />Votes: 2<br />Dem: 62.2%<br />Rep: 37.8%","MD<br />Votes: 10<br />Dem: 70.1%<br />Rep: 29.9%","MA<br />Votes: 11<br />Dem: 69.7%<br />Rep: 30.3%","NY<br />Votes: 28<br />Dem: 63.1%<br />Rep: 36.9%","RI<br />Votes: 4<br />Dem: 61.4%<br />Rep: 38.6%","VT<br />Votes: 3<br />Dem: 70.2%<br />Rep: 29.8%"],"hoverinfo":["text","text","text","text","text","text","text","text","text","text","text"],"type":"scatter","name":"Strong D","textfont":{"color":"rgba(2,118,171,1)"},"error_y":{"color":"rgba(2,118,171,1)"},"error_x":{"color":"rgba(2,118,171,1)"},"line":{"color":"rgba(2,118,171,1)"},"xaxis":"x","yaxis":"y","frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.20000000000000001,"selected":{"opacity":1},"debounce":0},"shinyEvents":["plotly_hover","plotly_click","plotly_selected","plotly_relayout","plotly_brushed","plotly_brushing","plotly_clickannotation","plotly_doubleclick","plotly_deselect","plotly_afterplot","plotly_sunburstclick"],"base_url":"https://plot.ly"},"evals":[],"jsHooks":[]}</script>

Clearly, this would be a very unfortunate electoral college result for Vice President Kamala Harris, as she loses almost every single swing state.

# Quantifying uncertainty

To determine how uncertain my predictions are, we can run Monte Carlo simulations of the election. For the sake of simplicity for this blog post, we will only run the simulations at the state level, and we will assume the national vote margin is true. Our simulations rely on the fact that each state’s predicted vote margin is actually a normal random variable, with mean centered at the predicted value and a standard deviation of three percent. (Note: this value is arbitrary and hard-coded, but in future weeks we will find a way of endogenizing it, perhaps by using the square root of the variance in the state’s recent voting history as the standard deviation instead.)

Then, following the methodology from the [*Economist*](https://www.economist.com/interactive/us-2024-election/prediction-model/president/how-this-works), we run 10,001 election simulations, recording the total number of electoral college votes each candidate wins in each simulation.

The following graph plots smoothed histograms for the electoral college votes for Harris and Trump respectively

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-7-1.png" width="672" />
From these simulations, Harris wins approximately 1.4 percent of the time, and Trump wins approximately 97.6 percent of the time. (The remaining percent accounts for ties, when both candidates win 269 electoral votes.) Note that the curves plotting Harris’s electoral votes and Trump’s electoral votes are symmetric. This makes sense, because they must sum to 538.
