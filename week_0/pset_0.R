library(tidyverse)
library(ggplot2)
library(modelr)
options(na.action = na.warn)


# Question 1: Data import

state_data <- read_csv("data/state_2pv_1948_2020.csv")
national_data <- read_csv("data/nat_pv_1860_2020.csv")

# Question 2: Data cleaning

state_df <- state_data %>% 
  select(year, state, party, two_party_vote_share) %>% 
  arrange(desc(year))

national_df <- national_data %>% 
  mutate(
    dem_tpv = npv_democrat / (npv_democrat + npv_republican),
    rep_tpv = npv_republican / (npv_democrat + npv_republican)
      ) %>% 
  filter(year >= 1948) %>% 
  pivot_longer(c(dem_tpv, rep_tpv), names_to = "party", values_to = "national_tpv") %>% 
  mutate(party = recode(party, "dem_tpv" = "Democrat", "rep_tpv" = "Republican")) %>% 
  select(-c("npv_democrat", "npv_republican"))

# Which state, California or Massachusetts, has been more Democratic since 2000?
state_df %>% 
  filter(year >= 2000, party == "Democrat") %>% 
  group_by(state) %>% 
  summarize(
    count = n(),
    avg_dem_tpv = mean(two_party_vote_share)
  ) %>% 
  filter(state == "California" | state == "Massachusetts")

# Question 3: Pivot to long
# *** Completed in Question 2 ***


# Question 4: Merge

merged <- state_df %>% 
  left_join(national_df, by = c("year", "party"))

# Question 5: Pivot to wide
merged_wide <- state_data %>% 
  select(year, state, party, two_party_vote_share) %>% 
  arrange(desc(year)) %>% 
  pivot_wider(names_from = c(state, party), values_from = two_party_vote_share, names_sep = "_") %>%
  left_join(national_data %>% 
              mutate(
                dem_tpv = 100 * (npv_democrat / (npv_democrat + npv_republican)),
                rep_tpv = 100 * (npv_republican / (npv_democrat + npv_republican))
              ) %>% 
              filter(year >= 1948) %>% 
              select(-c("npv_democrat", "npv_republican")),
            by = "year"
  )

# Question 6: Two basic linear models
model_1 <- lm(dem_tpv ~ Florida_Democrat, merged_wide)
summary(model_1)

model_2 <- lm(dem_tpv ~ `Florida_Democrat` + `New York_Democrat`, merged_wide)
summary(model_2)

# Question 7: Prediction

newdata = tibble(`Florida_Democrat` = c(48), `New York_Democrat` = c(60))
predict(model_2, newdata = newdata, type = "response")

# Question 8: Write Data
write.csv(merged, "data/merged_long.csv")
write.csv(merged_wide, "data/merged_wide.csv")

# Question 9: Visualization
# a) Histogram

merged %>% 
  filter(party == "Democrat", state == "Florida") %>% 
  ggplot(mapping = aes(two_party_vote_share)) +
  geom_histogram(binwidth = 2)
  
# b) Scatterplot
merged %>% 
  filter(party == "Democrat", state == "Florida") %>% 
  ggplot(mapping = aes(two_party_vote_share, national_tpv, label = year)) +
  geom_point() +
  geom_label()


