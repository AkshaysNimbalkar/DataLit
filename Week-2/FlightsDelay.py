import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

flights = pd.read_csv("E:/DataScience_Study/3months/Data-Lit/week2 - Statistics/2.5-DataAnalysis/formatted_flights.csv")
flights.head(10)

flights.shape

plt.hist(flights['arr_delay'], color = 'blue', edgecolor = 'black', bins = int(180/5), alpha = 0.7)

sns.distplot(flights['arr_delay'], hist=True, kde=False, bins=int(180/5), color='blue', hist_kws={'edgecolor':'black'})

plt.title("Histogram of Arrival dealy")
plt.xlabel("Delay (min)")
plt.ylabel("Flights")


for i, binwidth in enumerate([1,5,10,15]):
    plt.subplot(2, 2, i+1)
    plt.hist(flights['arr_delay'], bins=int(180/binwidth), color='blue', edgecolor='black')
    plt.title("Histogram of with Binwidth = %d" %binwidth)
    plt.xlabel("Delay (min)")
    plt.ylabel("Flights")

plt.tight_layout()
plt.show()


# Histogram Fails to Readability: i.e if we want to show distribution for each flight

# All the overlapping bars make it nearly impossible to make comparisons between the airlines.

# Solutions:
# 1) side by side histogram
# Instead of overlapping the airline histograms, we can place them side-by-side

n1 = flights['name'].apply(lambda x: ''.join([w for w in x.split()]))
name_dict = {}
for n in n1:
    if n in name_dict:
        name_dict[n] = name_dict[n] + 1
    else:
        name_dict[n] = 1

import operator as op

name_dict_sorted = sorted(name_dict.items(), key=op.itemgetter(1), reverse=True)
top_five = dict(name_dict_sorted[:5])

x1 = list(flights[flights['name'] == 'United Air Lines Inc.']['arr_delay'])
x2 = list(flights[flights['name'] == 'JetBlue Airways']['arr_delay'])
x3 = list(flights[flights['name'] == 'ExpressJet Airlines Inc.']['arr_delay'])
x4 = list(flights[flights['name'] == 'Delta Air Lines Inc.']['arr_delay'])
x5 = list(flights[flights['name'] == 'American Airlines Inc.']['arr_delay'])

# Assign colors for each airline and the names
colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
airline_names = ['United Air Lines Inc.', 'JetBlue Airways', 'ExpressJet Airlines Inc.', 'Delta Air Lines Inc.', 'American Airlines Inc.']

# Make the histogram using a list of lists
# Normalize the flights and assign colors and names
plt.hist([x1, x2, x3, x4, x5], bins=int(180/5), normed=True, color=colors, label=airline_names)


# Plot formatting
plt.legend()
plt.xlabel('Delay (min)')
plt.ylabel('Normalized Flights')
plt.title('Side-by-Side Histogram with Multiple Airlines')

# but it is not effective graph too much info. to digest

# solution -2
# Instead of plotting the bars for each airline side-by-side,
# we can stack them by passing in the parameter stacked = True to the histogram call:

plt.hist([x1, x2, x3, x4, x5], bins=int(180/5), normed=True, stacked=True, color=colors, label=airline_names)

# Plot formatting
plt.legend()
plt.xlabel('Delay (min)')
plt.ylabel('Normalized Flights')
plt.title('Stacked Histogram with Multiple Airlines')

# this is alos not very clear
# For example, at a delay of -15 to 0 minutes, does United Air Lines or JetBlue Airlines have a larger size of the bar?

# SO will move to density plots

sns.distplot(flights['arr_delay'], hist=True, kde=True, bins=int(180/5), color='darkblue',
                                    hist_kws={'edgecolor':'black'},
                                    kde_kws={'linewidth':4})
plt.xlabel('Delay (min)')
plt.ylabel('Density')
plt.title('Density using Seaborn')

# The curve shows the density plot which is essentially a smooth version of the histogram.
# The y-axis is in terms of density,
# and the histogram is normalized by default so that it has the same y-scale as the density plot.

# Solution #3 Density Plot
# solve our problem of visualizing the arrival delays of multiple airlines.
# To show the distributions on the same plot, we can iterate through the airlines, each time calling distplot
# with the kernel density estimate set to True and the histogram set to False

# List of five airlines to plot
airlines = ['United Air Lines Inc.', 'JetBlue Airways', 'ExpressJet Airlines Inc.','Delta Air Lines Inc.', 'American Airlines Inc.']

# iterate through airlines

for airline in airlines:
    # subset to airline
    subset = flights[flights['name'] == airline]

    # draw density plot for each subset
    sns.distplot(subset['arr_delay'], kde=True, hist=False,  kde_kws={'linewidth':3}, label=airline)

# Plot formatting
plt.legend(prop={'size': 16}, title = 'Airline')
plt.title('Density Plot with Multiple Airlines')
plt.xlabel('Delay (min)')
plt.ylabel('Density')

# Now that we finally have the plot we want, \
# we come to the conclusion that all these airlines have nearly identical arrival delay distributions!


# For other airlines comparison
airlines = ['United Air Lines Inc.', 'Alaska Airlines Inc.']
for airline in airlines:
    subset = flights[flights['name'] == airline]
    sns.distplot(subset['arr_delay'], hist = False, kde = True,
                 kde_kws={'shade': True, 'linewidth': 3},
                  label = airline)
plt.legend(prop={'size': 16}, title = 'Airline')
plt.title('Density Plot with Multiple Airlines')
plt.xlabel('Delay (min)')
plt.ylabel('Density')

# we finally have some useful information: Alaska Airlines flights tend to be earlier more often than United Airlines.


# Rug Plot: If you want to show every value in a distribution and not just the smoothed density, you can add a rug plot.
# call of rug = True


airlines = ['Alaska Airlines Inc.']
for airline in airlines:
    subset = flights[flights['name'] == airline]
    sns.distplot(subset['arr_delay'], hist = False, kde = True, rug=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 rug_kws={'color':'black'},
                  label = airline)
plt.legend(prop={'size': 16}, title = 'Airline')
plt.title('Rug Plot with Alaska Airlines')
plt.xlabel('Delay (min)')
plt.ylabel('Density')


################################################# Which Airline we shud choose: ########################3

Airlines = flights['name'].unique()

for airline in Airlines:
    subset = flights[flights['name'] == airline]
    sns.distplot(subset['arr_delay'], hist=False, kde=True,
                    kde_kws={'linewidth':3},
                    label=airline)
plt.legend(prop={'size': 16}, title = 'Airline')
plt.title('Density Plot with Dealy Airlines')
plt.xlabel('Delay (min)')
plt.ylabel('Density')

# Alaska Air Lines Inc. wiil use as it has less delay around(-23 min) time compared to other airlines but only 689 points of data
# United Air Lines Inc. will also use as they have highest number pf data points 56359 and dealy time is around -13 min.
# So due to lage availability of data points and comparitivly low delay time will use United Air Lines Inc.

######################################################################################### PArt 2 #################################
import pandas as pd
from statistics import stdev
import seaborn as sns
import matplotlib.pyplot as plt



flights = pd.read_csv("E:/DataScience_Study/3months/Data-Lit/week2 - Statistics/2.5-DataAnalysis/formatted_flights.csv")

flights['arr_delay'].mean() # 1.2971
stdev(flights['arr_delay']) # 29.0644


sns.distplot(flights['arr_delay'], hist=True, kde=True, bins=int(180/5), color='blue',
             kde_kws={'linewidth':3},
             hist_kws={'edgecolor':'black'})

plt.title("Histogram of Arrival dealy")
plt.xlabel("Delay (min)")
plt.ylabel("Flights")

# Sample 25%

flights_25 = flights.sample(frac=.25)

flights_25['arr_delay'].mean() # 1.3544
stdev(flights_25['arr_delay']) # 29.0799


sns.distplot(flights_25['arr_delay'], hist=True, kde=True, bins=int(180/5), color='blue',
             kde_kws={'linewidth':3},
             hist_kws={'edgecolor':'black'})

plt.title("Histogram of Arrival dealy")
plt.xlabel("Delay (min)")
plt.ylabel("Flights")


# Sample 40%

flights_40 = flights.sample(frac=.40)

flights_40['arr_delay'].mean() # 1.2374
stdev(flights_40['arr_delay']) # 29.0466


sns.distplot(flights_40['arr_delay'], hist=True, kde=True, bins=int(180/5), color='blue',
             kde_kws={'linewidth':3},
             hist_kws={'edgecolor':'black'})

plt.title("Histogram of Arrival dealy")
plt.xlabel("Delay (min)")
plt.ylabel("Flights")


# Sample 70%

flights_70 = flights.sample(frac=.70)

flights_70['arr_delay'].mean() # 1.3001
stdev(flights_70['arr_delay']) # 29.0699


sns.distplot(flights_70['arr_delay'], hist=True, kde=True, bins=int(180/5), color='blue',
             kde_kws={'linewidth':3},
             hist_kws={'edgecolor':'black'})

plt.title("Histogram of Arrival dealy")
plt.xlabel("Delay (min)")
plt.ylabel("Flights")

# All in One Graph:
for i, samp in enumerate([flights,flights_25,flights_40,flights_70]):
    plt.subplot(2,2,i+1)
    sns.distplot(samp['arr_delay'], hist=False, kde=True, bins=int(180 / 5), color='blue',
                 kde_kws={'shade':True,'linewidth': 3},
                 hist_kws={'edgecolor': 'black'})
    plt.title("Histogram of Samples")
    plt.xlabel("Delay (min)")
    plt.ylabel("Flights")

plt.tight_layout()
