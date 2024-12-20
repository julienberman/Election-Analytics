---
title: Final Election Prediction
author: Package Build
date: '2024-10-30'
slug: final-election-prediction
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

## Overview

Election week has come! It is a very scary time. We are all biting our nails in anticipation of the result on Tuesday and, of course, in anticipation of the predictions from the Election Analytics class. This entire semester, I have been developing a three-stage model that predicts the election results, and I am excited to provide a brief overview of my methodology, as well as my final predictions and simulations.

First, let me just provide a brief overview of how the overall prediction itself works. There are four models in total: two models that predict the national-level vote share of the Democratic candidate (i.e. the Democratic popular vote) — each using a different set of data — one model that predicts the state-level vote shares of the Democratic candidate (i.e. the electoral college), and one final model that ensembles the two national-level vote shares.

All four of these submodels are “elastic net” regressions. These are quite similar to simple linear regressions, in which coefficients are computed using Ordinary Least Squares, except that there is a penalty term added to shrink the values of each coefficient. The elastic net also has the benefit of automatically conducting feature selection, which means that predictors that do not contribute to reducing the overal varince get zeroed out.

## The Fundamentals Model

The first popular vote model uses “economic fundamentals,” which are economic indicators that describe the state of the economy in the lead-up to the election, to predict the national-level Democratic vote share. Here is a list of the economic indicators that I used:
- Total non-farm jobs
- Personal consumption expenditure
- Real disposable personal income
- Inflation, as measured by the annual change in the Consumer Price Index
- The stock market, based on the closing value of the S&P 500
- The consumer sentiment index calculated by the University of Michigan
- The unemployment rate

The above variables provide a cohesive snapshot of the economy at any given point in time. The most important contribution here is the inclusion of consumer sentiment — a variable that many forecasters have not examined in the past. The reason I include it is because recently, consumer sentiment about the economy — i.e. how people think the economy is doing — is drastically different from the actual state of the economy (Radcliffe et. al. (2024)). There are many explanations for this “sentiment gap.” Perhaps its because of the recent interest rate hikes that increase the cost of borrowing (Summers et. al. (2024)). Perhaps it’s because the period from February 2021 to June 2022 was the first time consumers in the United States experienced prolonged high levels of inflation since the 1990s, after the oil price shock in response to Iraq’s invasion of Kuwait. Regardless, this divergence suggests that traditional economic indicators may not entirely capture voters’ underlying behavior, because they may have a warped understanding of the economy.

The reason that I create a fundamentals forecast at all is because several prominent political science papers have noticed that economic strength in the leadup to the election appears to be a fairly strong predictor of incumbent performance (see e.g. Achen and Bartels (2017)). The intuition for this is that voters may credit or blame the current leadership for the state of the economy, even when many economic factors are outside direct political control.

Two interesting methodological choices that I made in the fundamentals model are a) the method of data aggregation and b) the timeframe of aggregation. Most of these economic indicators are reported weekly since 1948. Given that presidential elections occur only once every four years, we therefore need some way to collapse all the data for a given cycle into a single observation. There are two main ways to do this: “rate aggregation” and “mean aggregation.” Rate aggregation measures the percent change in a given indicator between time `\(t_1\)` and time `\(t_2\)`. Mean aggregation measures the average value of an indicator for all observations between time `\(t_1\)` and time `\(t_2\)`. Both aggregation measures have their advantages, but I ultimately decided on rate aggregation. Because memory is short, opinions about the economy are likely better reflected based on the change in economic conditions, rather than the level.

There is also the question of the aggregation timeframe. Several classic models — like the “time for change” model — use quarter two of the election year as the timeframe (Abramowicz (2008)). However, I thought that this selection was a bit arbitrary, so I looped through all possible values of `\(t_1\)` and `\(t_2\)` and selected the time period that reduced the overall error of the model. Eventually, I settled on a period from 30 weeks out from the election to 8 weeks out from the election.

Importantly, because each of the variables affect the *incumbent*, and the dependent variable in my model is the *democratic vote share*, I needed to interact each economic indicator with a dummy variable for incumbency that equals `\(+1\)` if the Democratic party is the incumbent party and `\(-1\)` if the Republican party is the incumbent party.

## The Polls Model

In addition to the fundamentals model, I also used a model based on national-level polling data and the incumbent’s approval rating. For this model, I included a variable for the national polling average (published by 538) each week prior to the election from 30 weeks prior to the election up until election week.

The most interesting methodological decision here is actually how to calculate the approval rating. Like the economic indicators, there is a big question about how to aggregate this metric. Rather than doing change over time, though, for this metric I instead used a weighted average, where the approval ratings closer to the election were given greater weight than the older approval ratings. The weighting is performed using an exponential decay function, `\(a = e^{-kw}\)`, where `\(w\)` is the number of weeks out from the election and `\(k\)` is a parameter set to `\(0.01\)`.

Also, like the economic variables in the fundamentals model, the approval rating variable only predicts the electoral success of the incumbent, not of the Democratic Party specifically. As such, I interact the approval rating with the same incumbency dummy as above.

## The electoral college model

Next, let’s examine the electoral college model. Importantly, this model does not directly predict the Democratic vote share in each state. Rather, it predicts a metric called “partisan lean,” which measures how much more or less democratic a given state is than the national popular vote. So, for example, if the popular vote is `\(52\%\)` for the Democratic candidate and Alabama has a partisan lean of `\(-20\)`, then the Democratic vote share in Alabama is `\(32\%\)`. Clearly, though, it is easy to convert from partisan lean to vote share once a national popular vote prediction is calculated.

Unlike in the national model, in the polls model I did not separate out into fundamentals and polls — largely because I did not believe that local-level economic fundamentals would be particularly predictive of local vote share. The literature has found that national-level economic variables tend to be more predictive than local ones.

As a result, I only used a single model for the electoral college, which includes the following variables:
- A state’s partisan lean in the previous election cycle
- A state’s partisan lean two election cycles prior
- Home state advantage
- Resident state advantage
- A state’s elasticity
- The Cook Political Report prediction for a given state
- The polling lean for a given state

The first two variables are autoregressive. The thing that best predicts partisan lean in the current election is vote share in he previous election. The third and fourth variables are dummy variables that are non-zero if a candidate was born in a given state or currently resides in a given state.

The fifth variable, elasticity, is a measure of how variable a given state is over time. If a state votes very Democratic in one election but very Republican in the next, it would have a very high elasticity.

The sixth variable — which is actually a vector of variables — indicates the rating that the Cook Political Report gave that state. The Cook Political Report consists of a set of a experts that publishes regular election forecasts. The forecast for a state is one of “Solid R,” “Likely R,” “Lean R,” “Toss Up,” “Lean D,” “Likely D,” or “Solid D.”

The seventh variable is also a vector of variables consisting of the polling for a given state. This is the same as the polling in the popular vote model — one variable for the polling data each week in the lead up to the election — with one notable exception: the value of each variable is not the poll itself, but rather the *poll lean*, which is the difference between that week’s state-level polling average and that week’s national-level polling average.

Finally, in my electoral college prediction, I include the fact that Maine and Nebraska split their electoral votes. They each allocate two electoral votes to the winner of the state’s popular vote. Then, they allocate one electoral vote to the popular vote winner in each congressional district. Thus, in total, my forecasts predicts vote margin in 54 jurisdictions: the two congressional districts in Maine, the three congressional districts in Nebraska, the other 48 states, and Washington D.C.

## The ensemble model

The fourth and final model is simply a way of aggregating the two popular vote predictions. It is a linear regression that uses the vote share predictions from the two national models as regressors on the actual historical vote share. The coefficients for this model are also not calculated with OLS. Instead, the optimization is constrained so that the coefficients cannot be negative and that they must sum to `\(1\)`. After all, the interpretation of the coefficient is that they are relative weights for each of the two popular vote models.

In all, here are the four regression specifications all laid out at once:

$$
\mathrm{pl} = \beta_0 + \beta_1 \mathrm{pl\_lag1} + \beta_2 \mathrm{pl\_lag2} + \beta_3\mathrm{hsa} + \beta_4\mathrm{rsa} + \beta_5\mathrm{\epsilon} + \vec{\beta_6}\upsilon + \vec{\beta_7}\chi + \varepsilon
$$

$$
\mathrm{dem\_share\_nat} = \beta_0 + \mathrm{incumb} \times (\beta_1\mathrm{jobs} + \beta_2 \mathrm{pce} + \beta_3\mathrm{rdpi} + \beta_4\mathrm{cpi} + \beta_5\mathrm{ics} + \beta_6\mathrm{sp500} + \beta_7\mathrm{unemp}) + \varepsilon
$$

$$
 \mathrm{dem\_share\_nat} = \beta_0 + \beta_1 \mathrm{incumb} \times \mathrm{approval} + \vec{\beta_{2}}\chi + \varepsilon
$$

$$
\mathrm{dem\_share\_nat} = \beta_0 + \beta_1 \mathrm{Model\#1} + \beta_{2}\mathrm{Model\#2} + \varepsilon
$$

## Simulation

Once my models are trained, I am able to produce point estimates for the national Democratic popular vote as well as for all 54 jurisdictions in my electoral college model. They point estimates are the most likely result, but there is also some uncertainty on either side of the prediction. To quantify the spread of this uncertainty, I calculate the “standard error of the prediction,” otherwise known as the “forecast variance.” The math behind the forecast variance is a bit tricky to parse. But it essentially amounts to the sum of the variance in the measurement error plus the sum of the variance in the predictions.

Once I have a meausure of the variance of each prediction, I can then create hypothetical distributions for the election result in each jurisdiction. I assume that vote shares are normal random variables with means equal to the point estimates that my model spit out and variances determined by the forecast variance. Then, I can create a giant multinomial distribution where I can take 100,000 sample draws from all 54 jursidictions, plus the national popular vote, at once.

Then, for each draw, I calculate the total number of electors each candidate won.

## Results

Here are the results from my four models, listed above. First, take a look at the results from the hyperparameter tuning of the elastic net.

This first one is for the state-level model:

<img src="fig1.png" width="345" />

The second is for the national-level polls model:

<img src="fig2.png" width="346" />

And the third is for the national-level polls model:

<img src="fig3.png" width="348" />

Now, here are the regression results for the state-level model,
<img src="fig4.png" width="535" />

For the national-level polls model:

<img src="fig5.png" width="528" />

And for the national-level fundamentals model.

<img src="fig6.png" width="519" />

Note that in the national fundamentals model, many of the economic indicators were dropped by the elastic net, but the index of consumer sentiment remained. This suggests that my hypothesis was right: maybe we should be focusing more on consumer sentiment at this point in time, given that it has diverged from the ground truth of the economy.

Here is a plot of the residuals for the national-level predictions, for both the polls model and the fundamentals model. This measures the difference between my predicted values for popular vote and the actual popular vote levels.

<img src="fig7.png" width="800" />

Note that the polls did a much better job on the whole of predicting national popular vote. This makes sense: economic fundamentals are incredibly noisy, and we likely don’t have a large enough sample size of elections in order to distinguish signal from noise. Consistent with this fact, it is notable that the ensemble model actually calculated weights such that the polls model received 100 percent of the weight, whereas the fundamentals model received 0 percent. Thus, the “national level predictions” are functionally just the popular vote predictions from my polls.

While this result initially surprised me, given that a lot of folks assert that economic fundamentals are so central to election outcomes, it turns out that it is largely consistent with 538’s weights — which are 97 percent in favor of the polls model and just 3 percent in favor of the fundamentals.

Here is my final prediction for the electoral college:

<div class="plotly html-widget html-fill-item" id="htmlwidget-1" style="width:672px;height:480px;"></div>
<script type="application/json" data-for="htmlwidget-1">{"x":{"visdat":{"1760c62189db":["function () ","plotlyVisDat"],"1760c3e18c02":["function () ","data"]},"cur_data":"1760c3e18c02","attrs":{"1760c62189db":{"mode":"markers","x":{},"y":{},"marker":{"symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":{},"hoverinfo":"text","color":{},"colors":["#e48782","#f0bbb8","#fbeeed","#e5f3fd","#6ac5fe","#0276ab"],"alpha_stroke":1,"sizes":[10,100],"spans":[1,20],"type":"scatter"}},"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"title":{"text":"2024 Electoral College Prediction","x":0.5,"y":0.94999999999999996},"showlegend":true,"xaxis":{"domain":[0,1],"automargin":true,"showgrid":false,"zeroline":false,"showticklabels":false,"range":[-50,960],"title":""},"yaxis":{"domain":[0,1],"automargin":true,"showgrid":false,"zeroline":false,"showticklabels":false,"range":[0,692.82032302755078],"scaleanchor":"x","scaleratio":1,"title":""},"plot_bgcolor":"white","paper_bgcolor":"white","annotations":[{"text":"AL","x":520,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"AK","x":120,"y":467.65371804359677,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"AZ","x":80,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"AR","x":400,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"CA","x":80,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"CO","x":200,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"CT","x":880,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"DE","x":840,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"DC","x":720,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"FL","x":640,"y":103.92304845413263,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"GA","x":600,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"HI","x":0,"y":103.92304845413263,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"ID","x":160,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"IL","x":480,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"IN","x":560,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"IA","x":400,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"KS","x":320,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"KY","x":520,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"LA","x":360,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"ME","x":920,"y":467.65371804359677,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"ME_d2","x":880,"y":415.69219381653051,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MD","x":760,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MA","x":840,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MI","x":600,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MN","x":360,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MS","x":440,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MO","x":440,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"MT","x":200,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NE","x":280,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NE_d2","x":360,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NV","x":120,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NH","x":800,"y":415.69219381653051,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NJ","x":800,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NM","x":240,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NY","x":760,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"NC","x":560,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"ND","x":280,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"OH","x":640,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"OK","x":280,"y":155.88457268119893,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"OR","x":120,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"PA","x":720,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"RI","x":920,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"SC","x":640,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"SD","x":320,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"TN","x":480,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"TX","x":240,"y":103.92304845413263,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"UT","x":160,"y":207.84609690826525,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"VT","x":720,"y":415.69219381653051,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"VA","x":680,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"WA","x":80,"y":415.69219381653051,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"WV","x":600,"y":259.80762113533154,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"WI","x":440,"y":363.73066958946424,"showarrow":false,"font":{"size":12,"color":"black"}},{"text":"WY","x":240,"y":311.76914536239786,"showarrow":false,"font":{"size":12,"color":"black"}},{"x":0.5,"y":-0.10000000000000001,"text":"Democratic EVs: 277 | Republican EVs: 261","showarrow":false,"xref":"paper","yref":"paper","font":{"size":14}}],"hovermode":"closest"},"source":"A","config":{"modeBarButtonsToAdd":["hoverclosest","hovercompare"],"showSendToCloud":false},"data":[{"mode":"markers","x":[520,400,160,520,280,280,320,600,240],"y":[155.88457268119893,207.84609690826525,311.76914536239786,259.80762113533154,363.73066958946424,155.88457268119893,311.76914536239786,259.80762113533154,311.76914536239786],"marker":{"color":"rgba(228,135,130,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["AL<br />Votes: 9<br />Dem: 39.4%<br />Rep: 60.6%","AR<br />Votes: 6<br />Dem: 38.7%<br />Rep: 61.3%","ID<br />Votes: 4<br />Dem: 36.7%<br />Rep: 63.3%","KY<br />Votes: 8<br />Dem: 38.9%<br />Rep: 61.1%","ND<br />Votes: 3<br />Dem: 35.6%<br />Rep: 64.4%","OK<br />Votes: 7<br />Dem: 35.9%<br />Rep: 64.1%","SD<br />Votes: 3<br />Dem: 38.7%<br />Rep: 61.3%","WV<br />Votes: 4<br />Dem: 33.5%<br />Rep: 66.5%","WY<br />Votes: 3<br />Dem: 31.1%<br />Rep: 68.9%"],"hoverinfo":["text","text","text","text","text","text","text","text","text"],"type":"scatter","name":"Strong R","textfont":{"color":"rgba(228,135,130,1)"},"error_y":{"color":"rgba(228,135,130,1)"},"error_x":{"color":"rgba(228,135,130,1)"},"line":{"color":"rgba(228,135,130,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[560,320,360,440,440,200,280,480,160],"y":[311.76914536239786,207.84609690826525,155.88457268119893,155.88457268119893,259.80762113533154,363.73066958946424,259.80762113533154,207.84609690826525,207.84609690826525],"marker":{"color":"rgba(240,187,184,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["IN<br />Votes: 11<br />Dem: 43.1%<br />Rep: 56.9%","KS<br />Votes: 6<br />Dem: 43.2%<br />Rep: 56.8%","LA<br />Votes: 8<br />Dem: 42.4%<br />Rep: 57.6%","MS<br />Votes: 6<br />Dem: 43.3%<br />Rep: 56.7%","MO<br />Votes: 10<br />Dem: 43.4%<br />Rep: 56.6%","MT<br />Votes: 4<br />Dem: 42.7%<br />Rep: 57.3%","NE<br />Votes: 2<br />Dem: 41.3%<br />Rep: 58.7%","TN<br />Votes: 11<br />Dem: 40.2%<br />Rep: 59.8%","UT<br />Votes: 6<br />Dem: 41.2%<br />Rep: 58.8%"],"hoverinfo":["text","text","text","text","text","text","text","text","text"],"type":"scatter","name":"Likely R","textfont":{"color":"rgba(240,187,184,1)"},"error_y":{"color":"rgba(240,187,184,1)"},"error_x":{"color":"rgba(240,187,184,1)"},"line":{"color":"rgba(240,187,184,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[120,80,640,600,400,560,640,640,240],"y":[467.65371804359677,207.84609690826525,103.92304845413263,155.88457268119893,311.76914536239786,207.84609690826525,311.76914536239786,207.84609690826525,103.92304845413263],"marker":{"color":"rgba(251,238,237,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["AK<br />Votes: 3<br />Dem: 45.2%<br />Rep: 54.8%","AZ<br />Votes: 11<br />Dem: 49.7%<br />Rep: 50.3%","FL<br />Votes: 30<br />Dem: 48.8%<br />Rep: 51.2%","GA<br />Votes: 16<br />Dem: 49.5%<br />Rep: 50.5%","IA<br />Votes: 6<br />Dem: 46.7%<br />Rep: 53.3%","NC<br />Votes: 16<br />Dem: 49.3%<br />Rep: 50.7%","OH<br />Votes: 17<br />Dem: 46.2%<br />Rep: 53.8%","SC<br />Votes: 9<br />Dem: 45.1%<br />Rep: 54.9%","TX<br />Votes: 40<br />Dem: 46.9%<br />Rep: 53.1%"],"hoverinfo":["text","text","text","text","text","text","text","text","text"],"type":"scatter","name":"Lean R","textfont":{"color":"rgba(251,238,237,1)"},"error_y":{"color":"rgba(251,238,237,1)"},"error_x":{"color":"rgba(251,238,237,1)"},"line":{"color":"rgba(251,238,237,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[200,920,880,600,360,360,120,800,240,720,680,440],"y":[259.80762113533154,467.65371804359677,415.69219381653051,363.73066958946424,363.73066958946424,259.80762113533154,259.80762113533154,415.69219381653051,207.84609690826525,311.76914536239786,259.80762113533154,363.73066958946424],"marker":{"color":"rgba(229,243,253,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["CO<br />Votes: 10<br />Dem: 54.8%<br />Rep: 45.2%","ME<br />Votes: 2<br />Dem: 53.3%<br />Rep: 46.7%","ME_d2<br />Votes: 1<br />Dem: 51.7%<br />Rep: 48.3%","MI<br />Votes: 15<br />Dem: 51.0%<br />Rep: 49.0%","MN<br />Votes: 10<br />Dem: 52.8%<br />Rep: 47.2%","NE_d2<br />Votes: 1<br />Dem: 51.7%<br />Rep: 48.3%","NV<br />Votes: 6<br />Dem: 51.1%<br />Rep: 48.9%","NH<br />Votes: 4<br />Dem: 52.7%<br />Rep: 47.3%","NM<br />Votes: 5<br />Dem: 54.6%<br />Rep: 45.4%","PA<br />Votes: 19<br />Dem: 50.5%<br />Rep: 49.5%","VA<br />Votes: 13<br />Dem: 54.1%<br />Rep: 45.9%","WI<br />Votes: 10<br />Dem: 50.5%<br />Rep: 49.5%"],"hoverinfo":["text","text","text","text","text","text","text","text","text","text","text","text"],"type":"scatter","name":"Lean D","textfont":{"color":"rgba(229,243,253,1)"},"error_y":{"color":"rgba(229,243,253,1)"},"error_x":{"color":"rgba(229,243,253,1)"},"line":{"color":"rgba(229,243,253,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[880,840,480,800,120,920,80],"y":[311.76914536239786,259.80762113533154,311.76914536239786,311.76914536239786,363.73066958946424,363.73066958946424,415.69219381653051],"marker":{"color":"rgba(106,197,254,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["CT<br />Votes: 7<br />Dem: 57.8%<br />Rep: 42.2%","DE<br />Votes: 3<br />Dem: 57.2%<br />Rep: 42.8%","IL<br />Votes: 19<br />Dem: 57.6%<br />Rep: 42.4%","NJ<br />Votes: 14<br />Dem: 56.8%<br />Rep: 43.2%","OR<br />Votes: 8<br />Dem: 56.6%<br />Rep: 43.4%","RI<br />Votes: 4<br />Dem: 58.4%<br />Rep: 41.6%","WA<br />Votes: 12<br />Dem: 58.2%<br />Rep: 41.8%"],"hoverinfo":["text","text","text","text","text","text","text"],"type":"scatter","name":"Likely D","textfont":{"color":"rgba(106,197,254,1)"},"error_y":{"color":"rgba(106,197,254,1)"},"error_x":{"color":"rgba(106,197,254,1)"},"line":{"color":"rgba(106,197,254,1)"},"xaxis":"x","yaxis":"y","frame":null},{"mode":"markers","x":[80,720,0,760,840,760,720],"y":[311.76914536239786,207.84609690826525,103.92304845413263,259.80762113533154,363.73066958946424,363.73066958946424,415.69219381653051],"marker":{"color":"rgba(2,118,171,1)","symbol":"hexagon","size":40,"line":{"color":"white","width":1}},"text":["CA<br />Votes: 54<br />Dem: 64.6%<br />Rep: 35.4%","DC<br />Votes: 3<br />Dem: 87.1%<br />Rep: 12.9%","HI<br />Votes: 4<br />Dem: 63.4%<br />Rep: 36.6%","MD<br />Votes: 10<br />Dem: 63.4%<br />Rep: 36.6%","MA<br />Votes: 11<br />Dem: 63.7%<br />Rep: 36.3%","NY<br />Votes: 28<br />Dem: 60.5%<br />Rep: 39.5%","VT<br />Votes: 3<br />Dem: 64.5%<br />Rep: 35.5%"],"hoverinfo":["text","text","text","text","text","text","text"],"type":"scatter","name":"Strong D","textfont":{"color":"rgba(2,118,171,1)"},"error_y":{"color":"rgba(2,118,171,1)"},"error_x":{"color":"rgba(2,118,171,1)"},"line":{"color":"rgba(2,118,171,1)"},"xaxis":"x","yaxis":"y","frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.20000000000000001,"selected":{"opacity":1},"debounce":0},"shinyEvents":["plotly_hover","plotly_click","plotly_selected","plotly_relayout","plotly_brushed","plotly_brushing","plotly_clickannotation","plotly_doubleclick","plotly_deselect","plotly_afterplot","plotly_sunburstclick"],"base_url":"https://plot.ly"},"evals":[],"jsHooks":[]}</script>

Finally, my simulation of outcomes:

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-12-1.png" width="672" />

Note that the simulation distributions are, of course, symmetric. Also note that the mode of the distribution actually occurs at a point where Harris loses the overall election. This is counterbalanced by the multitude of simulations where she wins over 300 electoral colelge votes.

my simulations are quite similar to those of 538. It is interesting that my standard errors are so large — nearly 6 points at times, which suggests a confidence interval of over 20 points in size — but it matches what Elliott Morris calculated, so I suppose I am doing something correct.
