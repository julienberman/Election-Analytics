---
title: 'Blog Post 6: Simulation'
author: Package Build
date: '2024-10-12'
slug: blog-post-6
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

``` r
# define formulas
state_formula <- as.formula(paste("pl_d_2pv ~ pl_d_2pv_lag1 + pl_d_2pv_lag2 + hsa_adjustment +",
                                  "rsa_adjustment + elasticity +", 
                                  "cpr_solid_d + cpr_likely_d	+ cpr_lean_d +", 
                                  "cpr_toss_up + cpr_lean_r + cpr_likely_r	+ cpr_solid_r + ",
                                  paste0("poll_pv_lean_", 7:36, collapse = " + ")))

nat_fund_formula <- as.formula("d_2pv_nat ~ incumb_party:(jobs_agg + 
                                     pce_agg + rdpi_agg + cpi_agg + ics_agg + 
                                     sp500_agg + unemp_agg)")


nat_polls_formula <- as.formula(paste("d_2pv_nat ~ incumb_party:(weighted_avg_approval) + ", 
                                           paste0("poll_pv_nat_", 7:36, collapse = " + ")))

# Split data, using the 1972 subset
state_data <- split_state(df_subset_1972, 2024)
national_data <- split_national(df_subset_1972, 2024)

# Predict state elasticities
state_data$test <- predict_elasticity(state_data$train, state_data$test)

# Train models
state_model_info <- train_elastic_net_fast(state_data$train, state_formula)
nat_fund_model_info <- train_elastic_net_fast(national_data$train, nat_fund_formula)
nat_polls_model_info <- train_elastic_net_fast(national_data$train, nat_polls_formula)
ensemble <- train_ensemble(list(nat_fund_model_info, nat_polls_model_info))

# Make predictions
state_predictions <- make_prediction(state_model_info, state_data$test)
nat_fund_predictions <- make_prediction(nat_fund_model_info, national_data$test)
nat_polls_predictions <- make_prediction(nat_polls_model_info, national_data$test)
# ensemble_predictions <- make_ensemble_prediction(ensemble, national_data$test)

# Create the prediction tibble
df_2024 <- tibble(
  state = state_data$test$state,
  abbr = state_data$test$abbr,
  electors = state_data$test$electors,
  partisan_lean = as.vector(state_predictions)
  ) %>%
  # Add national predictions - using first value since they're the same for all states
  mutate(
    d_2pv_polls = first(as.vector(nat_polls_predictions)),
    d_2pv_fund = first(as.vector(nat_fund_predictions))
 #   d_2pv_ensemble = first(as.vector(ensemble_predictions))
  ) %>%
  # Calculate final margins and color categories
  mutate(
    d_2pv_final = partisan_lean + d_2pv_polls,
    d_pv = d_2pv_final,
    r_pv = 100 - d_2pv_final,
    category = case_when(
      d_pv > 60 ~ "Strong D",
      d_pv > 55 & d_pv < 60 ~ "Likely D",
      d_pv > 50 & d_pv < 55 ~ "Lean D",
      d_pv > 45 & d_pv < 50 ~ "Lean R",
      d_pv > 40 & d_pv < 45 ~ "Likely R",
      TRUE ~ "Strong R"
    ),
    # Convert to factor with specific ordering
    category = factor(
      category,
      levels = c("Strong R", "Likely R", "Lean R", "Lean D", "Likely D", "Strong D")
    ),
    # calculate electors that each party wins
    d_electors = sum(ifelse(category %in% c("Lean D", "Likely D", "Strong D"), electors, 0)),
    r_electors = sum(ifelse(category %in% c("Lean R", "Likely R", "Strong R"), electors, 0))
  ) %>% 
  # filter unnecessary districts
  filter(!abbr %in% c("ME_d1", "NE_d1", "NE_d3"))

# run simulation
election <- simulate_election(nat_polls_model_info, state_model_info, national_data$test, state_data$test)


write_csv(df_2024, "df_2024.csv")
```

# Overview

Okay, something is seriously wrong with my model. Recall that my final model’s predictions involve an ensemble between the fundamentals model and the polling model. Well, this week when I ran the model, I realized that the ensemble assigned a *negative* weight to the fundamentals model. Admittedly it was only slightly negative, but still — that shouldn’t happen. Immediately, alarm bells went off. First of all, I modified the ensemble model from performing ordinary least squares, your classic regression, to non-negative least squares, which imposes an added restriction that all coefficients must be non-negative. I solved the optimization problem using [this](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1099-128X(199709/10)11:5%3C393::AID-CEM483%3E3.0.CO;2-L) method. Still, though, the ensemble model shrank the fundamentals model to zero, and assigned the polls model a weight of `\(1\)`.

Odd.

I confess, I spent a lot of time trying to find out why the fundamentals model was so bad, and I couldn’t entirely figure it out. I tried multiple different permutations of the regression specification, I tried aggregating the economic fundamentals a different way, I tried adding the base terms to the regression in addition to the cross terms. Nothing worked. Consistently, I would get `\(R^2\)` as low as `\(0.3\)`, and an adjusted `\(R^2\)` in the negatives. The out-of-sample MSE was often over four times as large as the in-sample MSE. And worst of all, the expected sign on the coefficients were the wrong direction — somehow, real disposable personal income was a negative predictor of vote margin, not a positive one.

Perhaps I will figure out what’s up with the fundamentals in a future week. In the mean time, it doesn’t actually affect my predictions that much — the polls model seems to be quite accurate:
- `\(R^2 = 0.879\)`
- `\(R_{\mathrm{adj}}^{2} = 0.793\)`
- `\(MSE_{\mathrm{out}} = 4.8\)`

One nice thing about this bug is that it actually inspired me to rerun the entire model, except instead of vote margin, I used vote share. While it didn’t solve the issue with the fundamentals model, it did reveal something: the vote share model is actually more accurate than the vote margin one! Moving forward, I will be using the vote share model instead. (This also has the added benefit of inadvertently resolving my Washington D.C. vote share issue from last week.)

I also fixed two other small mistakes. First, I realized that I was calculating a state’s elasticity incorrectly. Rather than measuring it in absolute terms, I changed it to be in relative terms:

$$
\varepsilon_t = \frac{\frac{s_{t} - s_{t-1}}{s_{t-1}}}{\frac{n_{t} - n_{t-1}}{n_{t-1}}}
$$
where `\(s_{t}\)` is the state’s vote in year `\(t\)` and `\(n_{t}\)` is the national vote in year `\(t\)`.

Second, I noticed that my state level forecast for the partisan lead included significant coefficients for several of Cook Political Report forecasts metics. However, I had neglected to include these forecasts in the testing data, so my results were somewhat biased. All these changes came together to give me the following map of the 2024 election:

<div class="plotly html-widget html-fill-item" id="htmlwidget-1" style="width:672px;height:480px;"></div>
<script type="application/json" data-for="htmlwidget-1">{"x":{"visdat":{"64156c5bb732":["function () ","plotlyVisDat"],"641579586b31":["function () ","data"]},"cur_data":"641579586b31","attrs":{"64156c5bb732":{"mode":"markers","x":{},"y":{},"marker":{"symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":{},"hoverinfo":"text","color":{},"colors":["#e48782","#f0bbb8","#fbeeed","#e5f3fd","#6ac5fe","#0276ab"],"alpha_stroke":1,"sizes":[10,100],"spans":[1,20],"type":"scatter"}},"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"title":{"text":"2024 Electoral College Prediction","x":0.5,"y":0.94999999999999996},"showlegend":true,"xaxis":{"domain":[0,1],"automargin":true,"showgrid":false,"zeroline":false,"showticklabels":false,"range":[-50,960],"title":""},"yaxis":{"domain":[0,1],"automargin":true,"showgrid":false,"zeroline":false,"showticklabels":false,"range":[0,692.82032302755078],"scaleanchor":"x","scaleratio":1,"title":""},"plot_bgcolor":"white","paper_bgcolor":"white","annotations":[{"text":"AL","x":520,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"AK","x":120,"y":467.65371804359677,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"AZ","x":80,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"AR","x":400,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"CA","x":80,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"CO","x":200,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"CT","x":880,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"DE","x":840,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"DC","x":720,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"FL","x":640,"y":103.92304845413263,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"GA","x":600,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"HI","x":0,"y":103.92304845413263,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"ID","x":160,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"IL","x":480,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"IN","x":560,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"IA","x":400,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"KS","x":320,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"KY","x":520,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"LA","x":360,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"ME","x":920,"y":467.65371804359677,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"ME_d2","x":880,"y":415.69219381653051,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MD","x":760,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MA","x":840,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MI","x":600,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MN","x":360,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MS","x":440,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MO","x":440,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MT","x":200,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NE","x":280,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NE_d2","x":360,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NV","x":120,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NH","x":800,"y":415.69219381653051,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NJ","x":800,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NM","x":240,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NY","x":760,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NC","x":560,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"ND","x":280,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"OH","x":640,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"OK","x":280,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"OR","x":120,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"PA","x":720,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"RI","x":920,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"SC","x":640,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"SD","x":320,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"TN","x":480,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"TX","x":240,"y":103.92304845413263,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"UT","x":160,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"VT","x":720,"y":415.69219381653051,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"VA","x":680,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"WA","x":80,"y":415.69219381653051,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"WV","x":600,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"WI","x":440,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"WY","x":240,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"x":0.5,"y":-0.10000000000000001,"text":"Democratic EVs: 288 | Republican EVs: 250","showarrow":false,"xref":"paper","yref":"paper","font":{"size":14}}],"hovermode":"closest"},"source":"A","config":{"modeBarButtonsToAdd":["hoverclosest","hovercompare"],"showSendToCloud":false},"data":[{"mode":"markers","x":[520,400,160,520,280,280,320,600,240],"y":[155.88457268119893,207.84609690826525,311.76914536239786,259.80762113533154,363.73066958946424,155.88457268119893,311.76914536239786,259.80762113533154,311.76914536239786],"marker":{"color":"rgba(228,135,130,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["AL<br />Votes: 9<br />Dem: 39.7%<br />Rep: 60.3%","AR<br />Votes: 6<br />Dem: 39.1%<br />Rep: 60.9%","ID<br />Votes: 4<br />Dem: 37.0%<br />Rep: 63.0%","KY<br />Votes: 8<br />Dem: 39.2%<br />Rep: 60.8%","ND<br />Votes: 3<br />Dem: 35.9%<br />Rep: 64.1%","OK<br />Votes: 7<br />Dem: 36.2%<br />Rep: 63.8%","SD<br />Votes: 3<br />Dem: 39.0%<br />Rep: 61.0%","WV<br />Votes: 4<br />Dem: 33.9%<br />Rep: 66.1%","WY<br />Votes: 3<br />Dem: 31.4%<br />Rep: 68.6%"],"hoverinfo":["text","text","text","text","text","text","text","text","text"],"type":"scatter","name":"Strong R","textfont":{"color":"rgba(228,135,130,1)"},"error_y":{"color":"rgba(228,135,130,1)"},"error_x":{"color":"rgba(228,135,130,1)"},"line":{"color":"rgba(228,135,130,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[560,320,360,440,440,200,280,480,160],"y":[311.76914536239786,207.84609690826525,155.88457268119893,155.88457268119893,259.80762113533154,363.73066958946424,259.80762113533154,207.84609690826525,207.84609690826525],"marker":{"color":"rgba(240,187,184,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["IN<br />Votes: 11<br />Dem: 43.5%<br />Rep: 56.5%","KS<br />Votes: 6<br />Dem: 43.6%<br />Rep: 56.4%","LA<br />Votes: 8<br />Dem: 42.8%<br />Rep: 57.2%","MS<br />Votes: 6<br />Dem: 43.7%<br />Rep: 56.3%","MO<br />Votes: 10<br />Dem: 43.7%<br />Rep: 56.3%","MT<br />Votes: 4<br />Dem: 43.1%<br />Rep: 56.9%","NE<br />Votes: 2<br />Dem: 41.7%<br />Rep: 58.3%","TN<br />Votes: 11<br />Dem: 40.5%<br />Rep: 59.5%","UT<br />Votes: 6<br />Dem: 41.5%<br />Rep: 58.5%"],"hoverinfo":["text","text","text","text","text","text","text","text","text"],"type":"scatter","name":"Likely R","textfont":{"color":"rgba(240,187,184,1)"},"error_y":{"color":"rgba(240,187,184,1)"},"error_x":{"color":"rgba(240,187,184,1)"},"line":{"color":"rgba(240,187,184,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[120,640,600,400,560,640,640,240],"y":[467.65371804359677,103.92304845413263,155.88457268119893,311.76914536239786,207.84609690826525,311.76914536239786,207.84609690826525,103.92304845413263],"marker":{"color":"rgba(251,238,237,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["AK<br />Votes: 3<br />Dem: 45.5%<br />Rep: 54.5%","FL<br />Votes: 30<br />Dem: 49.2%<br />Rep: 50.8%","GA<br />Votes: 16<br />Dem: 49.9%<br />Rep: 50.1%","IA<br />Votes: 6<br />Dem: 47.0%<br />Rep: 53.0%","NC<br />Votes: 16<br />Dem: 49.7%<br />Rep: 50.3%","OH<br />Votes: 17<br />Dem: 46.6%<br />Rep: 53.4%","SC<br />Votes: 9<br />Dem: 45.4%<br />Rep: 54.6%","TX<br />Votes: 40<br />Dem: 47.3%<br />Rep: 52.7%"],"hoverinfo":["text","text","text","text","text","text","text","text"],"type":"scatter","name":"Lean R","textfont":{"color":"rgba(251,238,237,1)"},"error_y":{"color":"rgba(251,238,237,1)"},"error_x":{"color":"rgba(251,238,237,1)"},"line":{"color":"rgba(251,238,237,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[80,920,880,600,360,360,120,800,240,720,680,440],"y":[207.84609690826525,467.65371804359677,415.69219381653051,363.73066958946424,363.73066958946424,259.80762113533154,259.80762113533154,415.69219381653051,207.84609690826525,311.76914536239786,259.80762113533154,363.73066958946424],"marker":{"color":"rgba(229,243,253,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["AZ<br />Votes: 11<br />Dem: 50.1%<br />Rep: 49.9%","ME<br />Votes: 2<br />Dem: 53.7%<br />Rep: 46.3%","ME_d2<br />Votes: 1<br />Dem: 52.1%<br />Rep: 47.9%","MI<br />Votes: 15<br />Dem: 51.4%<br />Rep: 48.6%","MN<br />Votes: 10<br />Dem: 53.1%<br />Rep: 46.9%","NE_d2<br />Votes: 1<br />Dem: 52.1%<br />Rep: 47.9%","NV<br />Votes: 6<br />Dem: 51.5%<br />Rep: 48.5%","NH<br />Votes: 4<br />Dem: 53.1%<br />Rep: 46.9%","NM<br />Votes: 5<br />Dem: 55.0%<br />Rep: 45.0%","PA<br />Votes: 19<br />Dem: 50.9%<br />Rep: 49.1%","VA<br />Votes: 13<br />Dem: 54.4%<br />Rep: 45.6%","WI<br />Votes: 10<br />Dem: 50.8%<br />Rep: 49.2%"],"hoverinfo":["text","text","text","text","text","text","text","text","text","text","text","text"],"type":"scatter","name":"Lean D","textfont":{"color":"rgba(229,243,253,1)"},"error_y":{"color":"rgba(229,243,253,1)"},"error_x":{"color":"rgba(229,243,253,1)"},"line":{"color":"rgba(229,243,253,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[200,880,840,480,800,120,920,80],"y":[259.80762113533154,311.76914536239786,259.80762113533154,311.76914536239786,311.76914536239786,363.73066958946424,363.73066958946424,415.69219381653051],"marker":{"color":"rgba(106,197,254,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["CO<br />Votes: 10<br />Dem: 55.2%<br />Rep: 44.8%","CT<br />Votes: 7<br />Dem: 58.2%<br />Rep: 41.8%","DE<br />Votes: 3<br />Dem: 57.6%<br />Rep: 42.4%","IL<br />Votes: 19<br />Dem: 57.9%<br />Rep: 42.1%","NJ<br />Votes: 14<br />Dem: 57.1%<br />Rep: 42.9%","OR<br />Votes: 8<br />Dem: 56.9%<br />Rep: 43.1%","RI<br />Votes: 4<br />Dem: 58.7%<br />Rep: 41.3%","WA<br />Votes: 12<br />Dem: 58.5%<br />Rep: 41.5%"],"hoverinfo":["text","text","text","text","text","text","text","text"],"type":"scatter","name":"Likely D","textfont":{"color":"rgba(106,197,254,1)"},"error_y":{"color":"rgba(106,197,254,1)"},"error_x":{"color":"rgba(106,197,254,1)"},"line":{"color":"rgba(106,197,254,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[80,720,0,760,840,760,720],"y":[311.76914536239786,207.84609690826525,103.92304845413263,259.80762113533154,363.73066958946424,363.73066958946424,415.69219381653051],"marker":{"color":"rgba(2,118,171,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["CA<br />Votes: 54<br />Dem: 64.7%<br />Rep: 35.3%","DC<br />Votes: 3<br />Dem: 87.4%<br />Rep: 12.6%","HI<br />Votes: 4<br />Dem: 63.7%<br />Rep: 36.3%","MD<br />Votes: 10<br />Dem: 63.8%<br />Rep: 36.2%","MA<br />Votes: 11<br />Dem: 64.0%<br />Rep: 36.0%","NY<br />Votes: 28<br />Dem: 60.8%<br />Rep: 39.2%","VT<br />Votes: 3<br />Dem: 64.8%<br />Rep: 35.2%"],"hoverinfo":["text","text","text","text","text","text","text"],"type":"scatter","name":"Strong D","textfont":{"color":"rgba(2,118,171,1)"},"error_y":{"color":"rgba(2,118,171,1)"},"error_x":{"color":"rgba(2,118,171,1)"},"line":{"color":"rgba(2,118,171,1)"},"xaxis":"x","yaxis":"y","frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.20000000000000001,"selected":{"opacity":1},"debounce":0},"shinyEvents":["plotly_hover","plotly_click","plotly_selected","plotly_relayout","plotly_brushed","plotly_brushing","plotly_clickannotation","plotly_doubleclick","plotly_deselect","plotly_afterplot","plotly_sunburstclick"],"base_url":"https://plot.ly"},"evals":[],"jsHooks":[]}</script>

A much better result for Harris, don’t you think?

# A New Simulation Method

My biggest improvement this week, however, was in the simulation process.

Simulations are used to test the uncertainty in an election prediction model. Running simulations allows me to introduce a slight variations in the values of my estimates and see just how much the election outcome changes as a result of these variations. Last week, I implemented an incredibly crude simulation approach. I assumed that the predicted national vote share was absolutely true, and that the predicted vote at the state level was the mean of a normal random variable with a standard deviation of three points.

Both of these assumptions are questionable. My prediction for the national vote share also has some degree of uncertainty, of course. Not to mention the fact that a three-point standard deviation was chosen completely arbitrarily. This week, I will introduce a new simulation process so that I avoid having to make the same assumptions I did last week.

For this simulation approach, I will introduce uncertainty in my *estimates*, not my *predictions*. This is an important distinction: the estimates refer to the vector of coefficients, `\(\vec{\beta}\)`, that I calculated using multiple regression. The predictions, on the other hand, refer to the vector of outputs, `\(\vec{\hat{y}}\)`, that I calculated by plugging in the 2024 values for all my predictors and computing a linear combination. I argue that varying `\(\vec{\beta}\)` makes more sense than varying `\(\vec{\hat{y}}\)`, because it accounts for the fact that sampling variability makes the model’s estimates of the underlying relationships between the different variables uncertain. The coefficients could be slightly different if I had different data, so by introducing variation in `\(\vec{\beta}\)`, I am capturing the true uncertainty in how predictors relate to the outcome.

So, I now simulate a set of coefficients by drawing from a multivariate normal distribution where the mean of each coefficient is the point estimate from the model, and the variance of each coefficient is determined by the square of its standard error. Of course, unlike with OLS regression, there is no easy closed form to get the variance of each coefficient. Instead, we have to “bootstrap” our model to calculate standard errors. Bootstrapping is a process where I draw `\(n\)` observations from my original dataset *with replacement* `\(s\)` different times. I will then make `\(s\)` different “copies” of my data. Then, I refit the model on each sample, and observe how the coefficients vary. By doing this, I can empirically estimate the variability in my coefficient estimates, and thus, the uncertainty in my election predictions.

Without further ado, here is the distribution:

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-4-1.png" width="672" />

Woah. Something is definitely wrong. First of all, it is good that the Republican and Democratic electoral college votes are symmetric. But the mode of the distribution should not be all the way in the 400s. After all, my point estimate indicated that Harris wins just 287 votes — enough to win, but not a landslide. Second, the distribution is quite wonky and uninterpretable. This is a problem, considering that it makes intuitive sense for closer elections to be more likely, and distant elections to be less so.

Ultimately, beyond coding mistakes (of which I am sure there are a few), I can think of several problems with my approach. First, my code, as written, introduces variation even in coefficients that the elastic net has shrunk to zero. These don’t bias my estimates, because the normal random variable would just be centered around zero, but they do introduce extranous noise into my simulation. Second, I am varying many coefficients at once. And not all these coefficients are statistically significant, which means they sometimes have a pretty high standard error. As a result, we see large swings in electoral college results, because the slight variances all get magnified. A slightly larger `\(\beta_1\)` gets applied, across the board, to all 50 states (and a few districts), which can end up having huge effects.

It seems that my prior was wrong, and that, in fact, it might be better to introduce variance somewhere else in the model. Next week, I will attempt to introduce variance in the testing data.