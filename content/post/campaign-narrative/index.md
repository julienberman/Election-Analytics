---
title: Campaign Narrative
author: Package Build
date: '2024-12-06'
slug: campaign-narrative
categories: []
tags: []
---

In my final post of the semester, I analyze why Michigan's 2024 presidential election results deviated from my pre-election forecast. First, I examine the state's demographics and electoral history. Then, I will compare the forecasted outcome with the actual results and examine specific campaign activities that may have contributed to the results.

# Demographics and Electoral History

Michigan was a crucial swing state during the 2024 election. Many polls conducted on the eve of the election, including 538's polling average, had the presidential race as a complete 50-50 toss up. The state swung between parties in the last three elections prior to 2024, supporting Obama in 2012, Trump in 2016, and Biden in 2020. Clearly, margins were razor thin. Ultimately, though, Trump won 49.7 percent of the vote, whereas Harris won just 48.3 percent.

Several key constituencies shape Michigan's political landscape. For one, the state has a substantial Arab American and MENA (Middle East and North Africa) population, particularly concentrated in areas like Dearborn, where nearly half of the 110,000 residents are of Arab descent. In addition, Detroit, home to America's car industry, has a significant union presence. The largest concentration of UAW members is in Michigan. Finally, Michigan, like many midwestern states, has large geographic divides. The state's major population centers include Detroit and its suburbs, Grand Rapids, and Ann Arbor. But Michigan also has large swaths of rural communities, too.

The deviation from pre-election forecasts can be attributed to several key factors that emerged during the campaign. Perhaps most significantly, there was a dramatic shift in Arab American voting patterns. Dearborn, which Biden had won by a nearly 3-to-1 margin in 2020, flipped to Trump in 2024. Trump received nearly 18,000 votes compared to Harris's 15,000 votes, marking the first Republican presidential victory in Dearborn since George W. Bush in 2000.

This shift was telegraphed months in advance. Local Democratic leaders had repeatedly warned party officials about growing dissatisfaction within the Arab American community regarding the administration's Middle East policy. The campaign's response to these concerns proved inadequate. While Harris attempted to moderate her rhetoric, she did not propose substantive policy changes. In contrast, Trump made direct overtures to the community, including a strategic visit to a Lebanese-owned restaurant in Dearborn days before the election. In addition, Harris lacked the strong support from unions that Biden had. Though the Michigan Teamsters Council and the UAW both supported her, two other major unions --- the International Association of Firefighters and International Brotherhood of Teamsters --- declined to endorse her, suggesting that blue collar workers, most without a college degree, might have been less inclined to vote Democrat.

Indeed, the divide among college-educated and non-college-educated voters was particularly stark in 2024. According to Matt Grossmann, a political scientist at Michigan State University, white college graduates overwhelmingly supported the Democrats, whereas the Republicans made gains across other demographic groups. Notably, the GOP saw improvements among Hispanic voters regardless of education level, and younger college voters showed movement toward Republicans.

Although the 2024 election saw lower turnout in general than the 2020 cycle, Michigan actually bucked the trend. Voters cast a record-breaking 5.7 million votes this year --- more than 100,000 more votes than in 2020. Notably, though, turnout shrank in eight out of the nine Democratic counties. Wayne County, the most-populated --- and most-important for Democrats --- had the largest vote decrease, with 13,899 fewer than 2002.

Michigan's down-ballot elections were also particularly contentious. For example, the open Senate seat created after Debbie Stabenow's retirement set up a fierce contest between Democrat Elissa Slotkin and Republican Mike Rogers. And despite Trump's victory at the top of the ticket, Slotkin narrowly prevailed with 48.64 percent of the vote to Rogers's 48.30 percent, suggesting that many Michigan voters engaged in ticket splitting.

# My Michigan Forecast

My forecast for Michigan predicted that Harris would have a two-party vote share of 51.01. Harris's actual two party vote share can be calculated as:

$$
dpv2p = \frac{dpv}{dpv + rpv}
$$
which gives a realized two party vote share of 49.28 percent. This means that my prediction was off by 1.73 percentage points.

Now, let's compare my prediction error for Michigan to the state level polling error. To do that, we simply take the average difference between the polls and the actual outcome. To ensure that we are measuring true polling error, I subset the data to only include polls in the fifteen weeks prior to the election. This calculation yields an average polling error of +1.98 percentage points. In other words, the polls on average overestimated Harris's margin by about two percentage points. This estimate for the polling error is fairly consistent with the national estimates.

The polling error is also quite similar to the error in my forecast, suggesting that my prediction was off in large part due to the fact that the polling may have systematically overestimated Harris's ability to win votes in Michigan. Thus, although my personal forecast did not closely match the actual results, if the polls had been more accurate, my model likely would have performed much better, largely because polls were the main ingredient in my model.

# Campaign specifics

Matthew provided a dataset in lab the other day with the text of the campaign speeches from both Harris and Trump. I wanted to see what sorts of things they talked about specifically in Michigan. Unfortunately, the speeches are not labelled by location, which means it is not always the easiest to tell which speeches are located where, but after skimming through them all, I was able to make an educated guess. Harris made four speeches in Michigan, and Trump made five.

Then, I performed sentiment analysis on the speeches. I used the Generalized Attribute Based Ratings Information Extraction Library (developed by my friends Elliott P. Mokski ’24 and Hemanth O. Asirvatham ’24), which is a programmatic and deterministic way to make ChatGPT rate text on attributes that the user inputs. I measured the following four attributes:
- discussion of economic issues
- discussion of domestic policy
- discussion of foreign policy
- discussino of cultural issues
(Note: the code for this was written in Python, which means that it is unfortunately unavailable for this blog post.)

These ratings are issued out of 100. Harris, across all her speeches, earned a 84 on economic issues, 72 on domestic policy, 27 on foreign policy, and 44 on cultural issues. In contrast, Trump earned a 73 on economic issues, a 50 on domestic policy, 62 on foreign policy, and a 93 on cultural issues.

These scorings, though imperfect, reveal distinct strategic emphases. Harris concentrated her message on: Economic policy details, manufacturing and jobs, and cost of living issues, whereas Trump emphasized cultural and identity issues such as immigration and border security. Perhaps Trump's cultural appeals resonated more with the lower-educated voters, where he made the biggest gains.


