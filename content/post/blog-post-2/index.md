---
title: 'Blog Post 2: Economic Fundamentals'
author: Package Build
date: '2024-09-11'
slug: blog-post-2
categories: []
tags: []
---

[SE = \sqrt{\frac{s_{1}^{2}}{n_{1}}+\frac{s_{2}^{2}}{n_{2}}}]

This week, I will expand the predictive model for the 2024 presidential election that I developed last week. In my previous model, I used the "partisan lean index" (PLI) --- which measures the difference between the state's democratic two-party vote share and the two party democratic vote share nationwide, and includes adjustments for home state advantage and state-level population density --- in the previous two election cycles to predict the electoral college results for the current election cycle.

Last week's model had a number of limitations. First, the outcome variable I was predicting --- two party vote share in each state --- does not actually determine who wins the state electors. Unfortunately, as much as I wish the likes of Jill Stein, Ralph Nader, and Cornell West didn't clutter up the ballot, the truth is that these third-party candidates, while rarely garnering more than a small fraction of the vote in any particular state, can have huge impacts on the overall state-wide election result. Consequently, this week, I plan to predict not two party vote _share_, but two party vote _margin_, a metric that a third-party candidate cannot distort.

Second, my previous forecast attempted to predict vote share in a single step using the PLI from the 2020 and 2016 elections. This time, I will add an intermediate step and use the 2020 and 2016 PLIs to forecast the 2024 PLI. Then, I will use the 2024 PLI, along with a multitude of other variables, to actually predict vote margin. This two-stage approach has two advantages. First, it allows me to more seamlessly integrate polling data later down the line, because I can easily create a national "snapshot" of the election by adding the PLI to the most current polling data that provides the nationwide vote margin. Second, it allows me to disaggregate the "politics" portion of the model from the "fundamentals" portion of the model. These two portions will then coalesce to produce my final prediction of vote margin.

Third, I improve my adjustments to the "partisan lean index." Now, I scale the home state advantage and resident state advantage adjustments by the size of the state, which is meant to capture the fact that candidates from smaller states tend to see larger effects. I also include a term that measures a state's elasticity, which captures the degree to which a given state "swings" from cycle to cycle. This adjustment only includes data from the 2008, 2012, 2016, and 2020 elections in order to most accurately capture the current political climate.

Fourth, I include the fact that Maine and Nebraska split their electoral votes. They each allocate two electoral votes to the winner of the state's popular vote. Then, they allocate one electoral vote to the popular vote winner in each congressional district. Thus, in total, my forecasts predicts vote margin in 54 jurisdictions: the two congressional districts in Maine, the three congressional districts in Nebraska, the other 48 states, and Washington D.C.

Finally, and most importantly, I construct from scratch the "fundamentals" forecast using the following six economic indicators:
* Total non-farm jobs
* Personal consumption expenditure
* Real disposable personal income
* Inflation, as measured by the annual change in the Consumer Price Index
* The stock market, based on the closing value of the S&P 500 
* The consumer sentiment index calculated by the University of Michigan
Each variable in the above set serves a particular function. 


_Inflation._ The period from February 2021 to June 2022 was the first time consumers in the United States experienced prolonged high levels of inflation since the 1990s after the oil price shock in response to Iraq's invasion of Kuwait.


```
## 
## % Table created by stargazer v.5.2.3 by Marek Hlavac, Social Policy Institute. E-mail: marek.hlavac at gmail.com
## % Date and time: Sun, Sep 15, 2024 - 23:10:20
## \begin{table}[!htbp] \centering 
##   \caption{OLS Regression Results for Economic Fundamentals Models} 
##   \label{} 
## \begin{tabular}{@{\extracolsep{5pt}}lcccccccc} 
## \\[-1.8ex]\hline 
## \hline \\[-1.8ex] 
##  & \multicolumn{8}{c}{\textit{Dependent variable:}} \\ 
## \cline{2-9} 
## \\[-1.8ex] & \multicolumn{8}{c}{National Vote Margin} \\ 
## \\[-1.8ex] & (1) & (2) & (3) & (4) & (5) & (6) & (7) & (8)\\ 
## \hline \\[-1.8ex] 
##  Jobs Growth & $-$18.450$^{***}$ & $-$5.441$^{***}$ &  &  &  &  &  &  \\ 
##   & (0.728) & (0.249) &  &  &  &  &  &  \\ 
##   & & & & & & & & \\ 
##  PCE Change & 9.577$^{***}$ &  & $-$5.073$^{***}$ &  &  &  &  &  \\ 
##   & (0.672) &  & (0.287) &  &  &  &  &  \\ 
##   & & & & & & & & \\ 
##  RDPI Change & 0.315 &  &  & 0.585 &  &  &  &  \\ 
##   & (0.410) &  &  & (0.360) &  &  &  &  \\ 
##   & & & & & & & & \\ 
##  Incumbency & $-$11.078$^{***}$ &  &  &  & $-$4.135$^{***}$ &  &  &  \\ 
##   & (0.527) &  &  &  & (0.279) &  &  &  \\ 
##   & & & & & & & & \\ 
##  ics\_agg & $-$1.835$^{***}$ &  &  &  &  & $-$2.563$^{***}$ &  &  \\ 
##   & (0.349) &  &  &  &  & (0.328) &  &  \\ 
##   & & & & & & & & \\ 
##  sp500\_agg & $-$3.247$^{***}$ &  &  &  &  &  & 3.740$^{***}$ &  \\ 
##   & (0.361) &  &  &  &  &  & (0.275) &  \\ 
##   & & & & & & & & \\ 
##  unemp\_agg & $-$5.981$^{***}$ &  &  &  &  &  &  & 3.912$^{***}$ \\ 
##   & (0.689) &  &  &  &  &  &  & (0.279) \\ 
##   & & & & & & & & \\ 
##  incumb & 12.518$^{***}$ & 7.488$^{***}$ & 5.112$^{***}$ & 6.043$^{***}$ & 7.670$^{***}$ & 5.117$^{***}$ & 6.676$^{***}$ & 7.425$^{***}$ \\ 
##   & (0.344) & (0.325) & (0.356) & (0.446) & (0.364) & (0.418) & (0.359) & (0.365) \\ 
##   & & & & & & & & \\ 
##  Constant & 0.744$^{***}$ & $-$0.351 & 0.417 & 0.541 & $-$0.321 & $-$0.196 & $-$0.484$^{*}$ & $-$0.361 \\ 
##   & (0.181) & (0.251) & (0.289) & (0.339) & (0.277) & (0.305) & (0.281) & (0.279) \\ 
##   & & & & & & & & \\ 
## \hline \\[-1.8ex] 
## Observations & 840 & 1,001 & 840 & 840 & 1,001 & 948 & 1,001 & 1,001 \\ 
## R$^{2}$ & 0.804 & 0.467 & 0.408 & 0.190 & 0.354 & 0.276 & 0.334 & 0.341 \\ 
## Adjusted R$^{2}$ & 0.803 & 0.466 & 0.407 & 0.188 & 0.352 & 0.275 & 0.333 & 0.339 \\ 
## Residual Std. Error & 4.775 (df = 831) & 7.766 (df = 998) & 8.276 (df = 837) & 9.680 (df = 837) & 8.550 (df = 998) & 9.111 (df = 945) & 8.677 (df = 998) & 8.635 (df = 998) \\ 
## F Statistic & 427.215$^{***}$ (df = 8; 831) & 436.596$^{***}$ (df = 2; 998) & 288.618$^{***}$ (df = 2; 837) & 98.325$^{***}$ (df = 2; 837) & 272.942$^{***}$ (df = 2; 998) & 180.551$^{***}$ (df = 2; 945) & 250.552$^{***}$ (df = 2; 998) & 257.808$^{***}$ (df = 2; 998) \\ 
## \hline 
## \hline \\[-1.8ex] 
## \textit{Note:}  & \multicolumn{8}{r}{$^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01} \\ 
## \end{tabular} 
## \end{table}
```


