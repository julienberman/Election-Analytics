---
title: Blog Post 1
author: Package Build
date: '2024-09-04'
slug: blog-post-1
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
## set working directory here
# setwd("~")

####----------------------------------------------------------#
#### Read and clean presidential popular vote.
####----------------------------------------------------------#

# Read presidential popular vote. 
d_popvote <- read_csv("data/popvote_1948-2020.csv")
```

```
## Rows: 38 Columns: 9
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## chr (2): party, candidate
## dbl (3): year, pv, pv2p
## lgl (4): winner, incumbent, incumbent_party, prev_admin
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

``` r
# Subset data to most recent past election year. 
d_popvote |> 
  filter(year == 2020) |> 
  select(party, candidate, pv2p)
```

```
## # A tibble: 2 × 3
##   party      candidate         pv2p
##   <chr>      <chr>            <dbl>
## 1 democrat   Biden, Joseph R.  52.3
## 2 republican Trump, Donald J.  47.7
```

``` r
# Pivot data to wide format with party names as columns and two-party vote share as values.
(d_popvote_wide <- d_popvote |>
    select(year, party, pv2p) |>
    pivot_wider(names_from = party, values_from = pv2p))
```

```
## # A tibble: 19 × 3
##     year democrat republican
##    <dbl>    <dbl>      <dbl>
##  1  1948     52.3       47.7
##  2  1952     44.7       55.3
##  3  1956     42.2       57.8
##  4  1960     50.1       49.9
##  5  1964     61.3       38.7
##  6  1968     49.6       50.4
##  7  1972     38.2       61.8
##  8  1976     51.1       48.9
##  9  1980     44.8       55.2
## 10  1984     40.9       59.1
## 11  1988     46.2       53.8
## 12  1992     53.6       46.4
## 13  1996     54.8       45.2
## 14  2000     50.3       49.7
## 15  2004     48.7       51.3
## 16  2008     53.8       46.2
## 17  2012     51.9       48.1
## 18  2016     51.2       48.8
## 19  2020     52.3       47.7
```

``` r
# Modify winner column to show "D" if Democrats win and "R" if Republicans win. 
(d_popvote_wide <- d_popvote_wide |> 
    mutate(winner = case_when(democrat > republican ~ "D",
                              TRUE ~ "R")))
```

```
## # A tibble: 19 × 4
##     year democrat republican winner
##    <dbl>    <dbl>      <dbl> <chr> 
##  1  1948     52.3       47.7 D     
##  2  1952     44.7       55.3 R     
##  3  1956     42.2       57.8 R     
##  4  1960     50.1       49.9 D     
##  5  1964     61.3       38.7 D     
##  6  1968     49.6       50.4 R     
##  7  1972     38.2       61.8 R     
##  8  1976     51.1       48.9 D     
##  9  1980     44.8       55.2 R     
## 10  1984     40.9       59.1 R     
## 11  1988     46.2       53.8 R     
## 12  1992     53.6       46.4 D     
## 13  1996     54.8       45.2 D     
## 14  2000     50.3       49.7 D     
## 15  2004     48.7       51.3 R     
## 16  2008     53.8       46.2 D     
## 17  2012     51.9       48.1 D     
## 18  2016     51.2       48.8 D     
## 19  2020     52.3       47.7 D
```

``` r
# Summarize data with respect to winners. 
d_popvote_wide |> 
  group_by(winner) |>
  summarise(races = n())
```

```
## # A tibble: 2 × 2
##   winner races
##   <chr>  <int>
## 1 D         11
## 2 R          8
```

``` r
####----------------------------------------------------------#
#### Visualize trends in national presidential popular vote. 
####----------------------------------------------------------#

# Visualize the two-party presidential popular over time. 
# TODO

####----------------------------------------------------------#
#### State-by-state map of presidential popular votes.
####----------------------------------------------------------#

# Sequester shapefile of states from `maps` library.
states_map <- map_data("state")

# Read wide version of dataset that can be used to compare candidate votes with one another. 
d_pvstate_wide <- read_csv("data/clean_wide_state_2pv_1948_2020.csv")
```

```
## Rows: 959 Columns: 14
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## chr  (1): state
## dbl (13): year, D_pv, R_pv, D_pv2p, R_pv2p, D_pv_lag1, R_pv_lag1, D_pv2p_lag...
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

``` r
# Merge d_pvstate_wide with state_map.
d_pvstate_wide$region <- tolower(d_pvstate_wide$state)

pv_map <- d_pvstate_wide |>
  filter(year == 2020) |>
  left_join(states_map, by = "region")

# Make map grid of state winners for each election year available in the dataset. 

# 2020 alone
pv_map %>% 
  mutate(winner = ifelse(D_pv > R_pv2p, "d", "r")) %>% 
  ggplot(aes(long, lat, group = group)) +
  geom_polygon(aes(fill = winner)) +
  scale_fill_manual(values = c("dodgerblue4", "red")) +
  theme_void()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-1-1.png" width="672" />

``` r
# all years, with scroller
d_pvstate_wide %>% 
  filter(year >= 1980) %>% 
  left_join(states_map, by = "region") %>% 
  mutate(winner = ifelse(D_pv > R_pv2p, "d", "r")) %>% 
  ggplot(aes(long, lat, group = group)) +
  facet_wrap(facets = year ~.) +
  geom_polygon(aes(fill = winner), color = 'white') +
  scale_fill_manual(values = c("dodgerblue4", "red")) +
  theme_void() +
  ggtitle("Presidential Vote Share (1980-2020)") +
  theme(strip.text = element_text(size = 12), aspect.ratio = 1)
```

```
## Warning in left_join(., states_map, by = "region"): Detected an unexpected many-to-many relationship between `x` and `y`.
## ℹ Row 1 of `x` matches multiple rows in `y`.
## ℹ Row 1 of `y` matches multiple rows in `x`.
## ℹ If a many-to-many relationship is expected, set `relationship =
##   "many-to-many"` to silence this warning.
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-1-2.png" width="672" />

``` r
####----------------------------------------------------------#
#### Forecast: simplified electoral cycle model. 
####----------------------------------------------------------#

# Create prediction (pv2p and margin) based on simplified electoral cycle model: 
# vote_2024 = 3/4*vote_2020 + 1/4*vote_2016 (lag1, lag2, respectively). 
pred_2024 <- d_pvstate_wide %>% 
  filter(year == 2020) %>% 
  left_join(states_map, by = "region") %>%
  mutate(
    d_vote_2024 = 3/4 * D_pv2p + 1/4*  D_pv2p_lag1,
    r_vote_2024 = 3/4 * R_pv2p + 1/4*  R_pv2p_lag1,
    winner_2024 = ifelse(d_vote_2024 > r_vote_2024, "d", "r"),
    margin_2024 = d_vote_2024 - r_vote_2024)

# plot who wins
pred_2024 %>% 
  ggplot(aes(long, lat, group = group)) +
  geom_polygon(aes(fill = winner_2024)) +
  scale_fill_manual(values = c("dodgerblue4", "red")) +
  theme_void()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-1-3.png" width="672" />

``` r
# plot margins
pred_2024 %>% 
  ggplot(aes(long, lat, group = group)) +
  geom_polygon(aes(fill = margin_2024, color = "black")) +
  scale_fill_gradient2(
   high = "dodgerblue4",
   low = "red",
   mid = "white",
   name = "win margin",
   breaks = c(-50, -25, 0, 25, 50),
   limits = c(-50, 50)) +
  theme_void()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-1-4.png" width="672" />

``` r
# Generate projected state winners and merge with electoral college votes to make 
# summary of electoral college vote distributions. 
# TODO

## Harris: 
## Trump: 
```
