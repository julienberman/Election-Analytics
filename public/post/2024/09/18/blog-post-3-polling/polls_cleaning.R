# polls cleaning

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

write_csv(polls_merged_v1, "polls_merged_v1.csv")

```


```{r}
# clean presidential approval
# Load all sheets from the Excel file
approval <- read_csv("data/approval.csv")

convert_dates <- function(df, column_name) {
  df %>%
    mutate(
      !!column_name := mdy(!!sym(column_name)),
      !!column_name := case_when(
        year(!!sym(column_name)) > 2040 ~ !!sym(column_name) - years(100),
        year(!!sym(column_name)) < 1940 ~ !!sym(column_name) + years(100),
        TRUE ~ !!sym(column_name)
      ),
      !!column_name := format(!!sym(column_name), "%m/%d/%Y")
    )
}
approval <- convert_dates(approval, "start_date")
approval <- convert_dates(approval, "end_date")
approval <- convert_dates(approval, "election_date")

approval <- approval %>% 
  group_by(election_date, weeks_left) %>% 
  summarize(
    start_date = first(start_date),
    end_date = first(end_date),
    approve = sum(approve * numeric_grade, na.rm = TRUE) / sum(numeric_grade, na.rm = TRUE),
    disapprove = sum(disapprove * numeric_grade, na.rm = TRUE) / sum(numeric_grade, na.rm = TRUE),
    party = first(party)
  ) %>% 
  ungroup() %>% 
  mutate(
    year = as.numeric(str_sub(election_date, -4)),
    weeks_left = -1 * weeks_left
  ) %>% 
  arrange(desc(year)) %>% 
  drop_na() %>% 
  filter(weeks_left <= 36)

approval <- approval %>%
  group_by(year) %>%
  summarize(
    weighted_avg_approval = sum(approve * exp(-0.01 * weeks_left)) / sum(exp(-0.01 * weeks_left))
  )

write_csv(approval, "data/approval_cleaned.csv")

approval_cleaned <- read_csv("data/approval_cleaned.csv")
polls_merged_v1 <- read_csv("polls_merged_v1.csv")

polls_merged_v2 <- polls_merged_v1 %>% 
  left_join(approval_cleaned, by="year") %>% 
  rename(
    approval = weighted_avg_approval
  ) %>% 
  mutate(
    poll_margin = d_poll_state - r_poll_state,
    poll_margin_nat = d_poll_nat - r_poll_nat
  )

write_csv(polls_merged_v2, "polls_merged_v2.csv")

# Pivot the data wider
polls_pivoted <- polls_merged_v2 %>%
  pivot_wider(
    id_cols = c(year, state, approval),
    names_from = weeks_left,
    values_from = c(poll_margin, poll_margin_nat),
    names_sort = TRUE
  )

write_csv(polls_pivoted, "polls_pivoted.csv")

# manually corrected NE-1 to Nebraska_d1, etc. Make sure to read from the csv

# process 2024 dataframe
df_2024 <- read_csv("data/data_2024.csv") 

# add the following:
# 1) popnorm, hsa, and rsa adjustments
# 2) partisan lean lags
# 3) elasticity lags

df_2024 <- df_2024 %>% 
  mutate(
    pop_norm = (pop - min(pop, na.rm = TRUE)) / (max(pop, na.rm = TRUE) - min(pop, na.rm = TRUE)),
    hsa_adjustment = ifelse(d_hsa == 1, 1 / pop_norm, ifelse(r_hsa == 1, -1 / pop_norm, 0)),
    rsa_adjustment = ifelse(d_rsa == 1, 1 / pop_norm, ifelse(r_rsa == 1, -1 / pop_norm, 0)),
    pl_lag1 = df_full %>% filter(year == 2020) %>% pull(pl),
    pl_lag2 = df_full %>% filter(year == 2020) %>% pull(pl_lag1),
    elasticity_lag1 = df_full %>% filter(year == 2020) %>% pull(elasticity),
    elasticity_lag2 = df_full %>% filter(year == 2020) %>% pull(elasticity_lag1)
  )

df_full <- df_full %>% mutate(electors = as.numeric(electors))

# combine with full dataset
df_combined <- bind_rows(df_full, df_2024)

# merge in polling data
polls <- read_csv("polls_pivoted.csv") %>% 
  select(-c("approval"))

merged_final <- df_combined %>% 
  left_join(approval, by = "year") %>% 
  left_join(polls, by = c("year", "state"))

write_csv(merged_final, "merged_v3.csv")

