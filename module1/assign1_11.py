#!/usr/bin/env python
"""
Please accomplish the instructions:
From the `Murder` data file ([http://stat4ds.rwth-aachen.de/data/Murder.dat](http://stat4ds.rwth-aachen.de/data/Murder.dat)) at the book’s website, use the variable murder, which is the murder rate (per 100,000 population) for each state in the U.S. in 2017 according to the FBI Uniform Crime Reports. At first, do not use the observation for D.C. (DC). Using software:
(a) Find the mean and standard deviation and interpret their values.
(b) Find the five-number summary, and construct the corresponding box plot. Interpret.
(c) Now include the observation for D.C. What is affected more by this outlier: The mean or the median? The range or the inter-quartile range?
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./murder.dat", sep="\s+")




breakpoint()
