#!/usr/bin/env python
"""
Exploratory Data Analysis (EDA) Class 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

# Make plot dark yo! ᕙ(▀̿ĺ̯▀̿ ̿)ᕗ
plt.style.use('dark_background')

def read_file(file):
    return pd.read_csv(file)

def missing_values(data):
    missing = data.isnull().sum().sum()

    return missing

def convert_data_numpy(dataframe):
    X = np.array(dataframe[dataframe.columns[:-1]])
    y = np.array(dataframe[dataframe.columns[-1]])

    return X, y

def plot_class_balance(data):
    # Frequencies of each class
    uniques, counts = np.unique(data, return_counts=True)

    # Plotting
    plt.figure(figsize=(8, 6))

    # Bar chart 
    plt.bar(x=uniques, height=counts, color="lime")

    plt.xlabel("?")
    plt.tight_layout()
    plt.grid(color="gray")
    plt.show()

def main():
    df = read_file("data/niblings.txt")

    # Basic data info
    print(f"\n{df.shape[0]} rows, and {df.shape[1]} columns\n")
    df.info()
    print("\n")
    # Since the example data is all categorical
    print(df.describe(include="all"))

    # Missing values
    missing = missing_values(df)
    print(f"\nMissing {missing} number of values\n")

    # Convert data to numpy arrays
    X, y = convert_data_numpy(df)

    # Bar chart
    plot_class_balance(y)
    
if __name__ == "__main__":
    main()
