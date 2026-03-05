#!/usr/bin/env python
"""
Analyze the `Carbon_West` ([http://stat4ds.rwth-aachen.de/data/Carbon_West.dat](http://stat4ds.rwth-aachen.de/data/Carbon_West.dat)) data file at the book’s website by **(a)** constructing a frequency distribution and  a histogram, **(b)** finding the mean, median, and standard deviation. Interpret each.
"""
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, log2

df = pd.read_csv("./carbon_west.dat", sep="\s+")

# Bins for continuous variable via Sturges' Rule: k = 1 + log2(n)
k = ceil(1 + log2(df.shape[0]))

# CO2 column
x = df["CO2"]
# Create bins into discrete intervals (continuous -> discrete variable)
bins = pd.cut(x, bins=k)

# Frequencies 
freqs = bins.value_counts().sort_index()
print(freqs)

# Proportions (for kicks)
print(f"Proprtions:\n{freqs/len(x)}")

# Plotting Histogram
plt.hist(x, bins=k, edgecolor="black")
plt.xlabel("CO2 Emissions (per tons per capita)")
plt.ylabel("Frequency")
plt.show()

# Output mean, median, and standard dev
mew = x.mean()
med = x.median()
std_dev = x.std()
print("\nStats:")
print(f"Mean: {mew:.3f}")
print(f"Median: {med:.3f}")
print(f"Standard Dev: {std_dev:.3f}")

# Interpretation
print("\nFrequency Distribution and Histogram:")
print("""
The histogram of the CO2 emissions is RIGHT-SKEWED, meaing most western
      countries have some moderate degree of CO2 emissions.  U.S., Canada and
      Australia have much higher values, extending the right tail.
      """)
print("\nMEAN:")
print("""
The mean CO2 emissions represent the average per-capita emission across all
  countries in the dataset.  Due to the right-skewness, the mean is yanked
  upward via the higher values, being larger than what would 'typical',
  here.
      """)
print("\nMEDIAN:")
print("""
Representing the 'middle' country when emisssions ordered from lowest to
highest, and is not effected by extreme values (outliers), and is better
for representing the 'typical' values for a skewed distribution.
      """)
print("\nSTANDARD DEVIATION:")
print("""
This measures the spread in CO2 emissions across the countries.  Larger values
      represent larger differences in emissions.
      """)

# Right skewed distribution
print(f"Mean = {mew:.3f} > {med:.3f} = Median")

breakpoint()








