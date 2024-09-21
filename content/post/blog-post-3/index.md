---
title: 'Blog Post 3: Polling'
author: Package Build
date: '2024-09-18'
slug: blog-post-3-polling
categories: []
tags: []
---






``` r
# Load libraries.
## install via `install.packages("name")`
library(ggplot2)
library(maps)
library(tidyverse)
```

```
## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
## ✔ dplyr     1.1.4     ✔ readr     2.1.5
## ✔ forcats   1.0.0     ✔ stringr   1.5.1
## ✔ lubridate 1.9.3     ✔ tibble    3.2.1
## ✔ purrr     1.0.2     ✔ tidyr     1.3.1
## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
## ✖ dplyr::filter() masks stats::filter()
## ✖ dplyr::lag()    masks stats::lag()
## ✖ purrr::map()    masks maps::map()
## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
```

``` r
library(usmap)
library(plotly)
```

```
## 
## Attaching package: 'plotly'
## 
## The following object is masked from 'package:ggplot2':
## 
##     last_plot
## 
## The following object is masked from 'package:stats':
## 
##     filter
## 
## The following object is masked from 'package:graphics':
## 
##     layout
```

``` r
library(gridExtra)
```

```
## 
## Attaching package: 'gridExtra'
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
```

``` r
library(sf)
```

```
## Linking to GEOS 3.11.0, GDAL 3.5.3, PROJ 9.1.0; sf_use_s2() is TRUE
```

``` r
library(ggrepel)
library(shiny)
library(leaflet)
library(stargazer)
```

```
## 
## Please cite as: 
## 
##  Hlavac, Marek (2022). stargazer: Well-Formatted Regression and Summary Statistics Tables.
##  R package version 5.2.3. https://CRAN.R-project.org/package=stargazer
```

``` r
library(blogdown)
library(car)
```

```
## Loading required package: carData
## 
## Attaching package: 'car'
## 
## The following object is masked from 'package:dplyr':
## 
##     recode
## 
## The following object is masked from 'package:purrr':
## 
##     some
```

``` r
library(lubridate)
library(zoo)
```

```
## 
## Attaching package: 'zoo'
## 
## The following objects are masked from 'package:base':
## 
##     as.Date, as.Date.numeric
```

``` r
library(modelsummary)
```

```
## `modelsummary` 2.0.0 now uses `tinytable` as its default table-drawing
##   backend. Learn more at: https://vincentarelbundock.github.io/tinytable/
## 
## Revert to `kableExtra` for one session:
## 
##   options(modelsummary_factory_default = 'kableExtra')
##   options(modelsummary_factory_latex = 'kableExtra')
##   options(modelsummary_factory_html = 'kableExtra')
## 
## Silence this message forever:
## 
##   config_modelsummary(startup_message = FALSE)
```

``` r
library(GGally)
```

```
## Registered S3 method overwritten by 'GGally':
##   method from   
##   +.gg   ggplot2
```

``` r
setwd("~/Documents/0 - Election Analytics/")
```



``` r
df_full <- read.csv("data/elec_full.csv")

# Read both dataframes in R
nat_polls <- read.csv("data/national_polls_1968-2024.csv")
state_polls <- read.csv("data/state_polls_1968-2024.csv")

state_polls_wider <- state_polls %>%
  pivot_wider(
    id_cols = c(year, state, weeks_left, days_left, poll_date), 
    names_from = party, 
    values_from = c(poll_state, before_convention),
  ) %>% 
  rename(
    "d_poll_state" = "poll_state_DEM",
    "r_poll_state" = "poll_state_REP",
    "d_convention" = "before_convention_DEM",
    "r_convention" = "before_convention_REP"
  ) %>% 
  mutate(
    d_poll_state = map_dbl(d_poll_state, ~ifelse(is.null(.x), NA_real_, as.numeric(.x))),
    r_poll_state = map_dbl(r_poll_state, ~ifelse(is.null(.x), NA_real_, as.numeric(.x))),
    d_convention = 1 - map_dbl(d_convention, ~ifelse(is.null(.x), NA_real_, as.numeric(.x))),
    r_convention = 1 - map_dbl(r_convention, ~ifelse(is.null(.x), NA_real_, as.numeric(.x))),
  )
```

```
## Warning: Values from `before_convention` and `poll_state` are not uniquely identified;
## output will contain list-cols.
## • Use `values_fn = list` to suppress this warning.
## • Use `values_fn = {summary_fun}` to summarise duplicates.
## • Use the following dplyr code to identify duplicates.
##   {data} |>
##   dplyr::summarise(n = dplyr::n(), .by = c(year, state, weeks_left, days_left,
##   poll_date, party)) |>
##   dplyr::filter(n > 1L)
```

``` r
nat_polls_wider <- nat_polls %>% 
  pivot_wider(
    id_cols = c(year, state, weeks_left, days_left, poll_date), 
    names_from = party, 
    values_from = c(poll_nat, before_convention)
  ) %>% 
  rename(
    "d_poll_nat" = "poll_nat_DEM",
    "r_poll_nat" = "poll_nat_REP",
    "d_convention" = "before_convention_DEM",
    "r_convention" = "before_convention_REP"
  ) %>% 
  mutate(
    d_poll_nat = map_dbl(d_poll_nat, ~ifelse(is.null(.x), NA_real_, as.numeric(.x))),
    r_poll_nat = map_dbl(r_poll_nat, ~ifelse(is.null(.x), NA_real_, as.numeric(.x))),
    d_convention = 1 - map_dbl(d_convention, ~ifelse(is.null(.x), NA_real_, as.numeric(.x))),
    r_convention = 1 - map_dbl(r_convention, ~ifelse(is.null(.x), NA_real_, as.numeric(.x))),
  )
```

```
## Warning: Values from `before_convention` and `poll_nat` are not uniquely identified;
## output will contain list-cols.
## • Use `values_fn = list` to suppress this warning.
## • Use `values_fn = {summary_fun}` to summarise duplicates.
## • Use the following dplyr code to identify duplicates.
##   {data} |>
##   dplyr::summarise(n = dplyr::n(), .by = c(year, state, weeks_left, days_left,
##   poll_date, party)) |>
##   dplyr::filter(n > 1L)
```

``` r
polls_merged_v1 <- state_polls_wider %>% 
  left_join(nat_polls_wider, by = c("year", "weeks_left", "days_left", "poll_date", "d_convention", "r_convention")) %>%
  select(-c("state.y")) %>% 
  rename(
    "state" = "state.x"
  ) %>% 
  group_by(year, weeks_left, state) %>% 
  summarize(
    d_poll_state = mean(d_poll_state, na.rm = TRUE),
    r_poll_state = mean(r_poll_state, na.rm = TRUE),
    d_poll_nat = mean(d_poll_nat, na.rm = TRUE),
    r_poll_nat = mean(r_poll_nat, na.rm = TRUE),
    r_convention = first(r_convention),
    d_convention = first(d_convention)
  ) %>%
  ungroup() %>% 
  arrange(desc(year), desc(weeks_left), state)
```

```
## `summarise()` has grouped output by 'year', 'weeks_left'. You can override
## using the `.groups` argument.
```

``` r
write_csv(polls_merged_v1, "polls_merged_v1.csv")
```



``` r
# clean presidential approval
```

