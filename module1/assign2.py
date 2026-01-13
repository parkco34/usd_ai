#!/usr/bin/env python
"""
## **Problem # 1.2.**
The `Students` data file at [http://stat4ds.rwth-aachen.de/data/Students.dat](http://stat4ds.rwth-aachen.de/data/Students.dat) responses of a class of 60
social science graduate students at the University of Florida to a questionnaire that asked about *gender* (1 = female, 0 = male), *age*, *hsgpa* = high school GPA (on a four-point scale), cogpa = college GPA, *dhome* = distance (in miles) of the campus from your home town, *dres* = distance (in miles) of the classroom from your current residence, *tv* = average number of hours per week that you watch TV, *sport* = average number of hours per week that you participate in sports or have other physical exercise, *news* = number of
times a week you read a newspaper, *aids* = number of people you know who have died from AIDS or who
are HIV+, *veg* = whether you are a vegetarian (1 = yes, 0 = no), *affil* = political affiliation (1 = Democrat, 2
= Republican, 3 = independent), *ideol* = political ideology (1 = very liberal, 2 = liberal, 3 = slightly liberal, 4
= moderate, 5 = slightly conservative, 6 = conservative, 7 = very conservative), *relig* = how often you
attend religious services (0 = never, 1 = occasionally, 2 = most weeks, 3 = every week), *abor* = opinion
about whether abortion should be legal in the first three months of pregnancy (1 = yes, 0 = no), *affirm* =
support affirmative action (1 = yes, 0 = no), and *life* = belief in life after death (1 = yes, 2 = no, 3 =
undecided). You will use this data file for some exercises in this book.

(a) Practice accessing a data file for statistical analysis with your software by going to the book’s
     website and copying and then displaying this data file.

(b) Using responses on *abor*, state a question that could be addressed with (i) descriptive
     statistics, (ii) inferential statistics.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./students.dat", sep="\s+")
print(f"Students stat summary:\n{df.describe()}")
print(df.info())

# Combine categorical variables for the purpose of ensuring appropriate
# combinations of variable values
df["affil_relig"] = df["affil"].astype(str) + "_" + df["relig"].astype(str)

# ----------------------------------------
# Claude for outsourcing print statements task
# to inform user of variable values
# ----------------------------------------
# Explain the categorical variables
print("\n" + "="*50)
print("VARIABLE DEFINITIONS:")
print("="*50)

print("\n'affil' - Political Affiliation:")
print("  1 = Democrat")
print("  2 = Independent")
print("  3 = Republican")

print("\n'relig' - Religious Service Attendance:")
print("  0 = Never")
print("  1 = Rarely (less than once a year)")
print("  2 = Occasionally (a few times a year)")
print("  3 = Frequently (weekly or more)")

print("\n'abor' - Support for Legal Abortion:")
print("  0 = No (oppose)")
print("  1 = Yes (support)")

print("\n" + "="*50)
print("READING THE CONTINGENCY TABLE:")
print("="*50)
print("Row labels (e.g., '1_2') represent: affil_relig")
print("  First number = political affiliation")
print("  Second number = religious attendance level")
print("Example: '2_3' = Independent who attends services frequently")
print("""('2_3', 0) -> 6 = ((independent, frequently going to church), opposes
      abortion) -> COUNT of students who oppose abortion""")

# =========== DESCRIPTIVE STATS ===========
DESC_QUESTION = """How does support for legal abortion vary across levels of religious
service attendance and political affliliation?"""
print("\nDescriptive Stats Question:\n")
print(f"{DESC_QUESTION}")

# Contingency table
freqs = pd.crosstab(index=df["affil_relig"], columns=df["abor"])

print("\nCONTINGENCY TABLE (Counts):")
print(freqs)
print("="*50 + "\n")

# PLotting data
# Proportions within each (affil, relig) group
props = freqs.div(freqs.sum(axis=1), axis=0)

props.plot(
    kind="bar",
    stacked=True,
    figsize=(15, 8)
)

plt.title("Proportion Supporting Abortion by Political Affiliation and Religious Attendance")
plt.xlabel("Political Affiliation _ Religious Attendance")
plt.ylabel("Proportion")
plt.legend(["Oppose", "Support"], title="Abortion")
plt.tight_layout()
plt.show()

# =========== INFERENTIAL STATS ===========
INFER_QUESTION = """
Is support for legal abortion statistically associated with
political affiliation and religious attendance in the population
of similar graduate students?
"""
print("\nInferntial Stats Question:")
print(f"{INFER_QUESTION}")
print("""\n
The SAMPLE is the 60 students.
The POPULATION is the broader population of Social Science grad students
      at the University of Florida.
The SAMPLE STATISTICS is the proportion of students who support legalized
      abortion w/in each combination of political affiliation and religious
      attendance.
The POPULATION PARAMETERS  being the true proportions of support for abortion
      w/in political affiliation and religious attendance; unknown.
      """)
print("""\nInterpretation:\n
HIgher religious attendance corresponds, generally, to lower support rates.
Lower levels of attendance tend towards being associated with higher support
      across affiliations.\n
The data comes from an OBSERVATIONAL STUDY instead of a randomized experiment,
      so there's no causal conclusion that can be drawn from the sample.
We also have a small sample size with limited subgroup counts.  This will
      likely increase UNCERTAINTY.
      """)


