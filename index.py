import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read in your dataset
df = pd.read_csv("data.csv")

# View the first few rows of the data
print(df.head())

print(df.describe())

# Perform preprocessing on the dataset
# This may include handling missing values, scaling/normalizing numerical data,
# and encoding categorical data
# Convert All Nan values to 0
df = df.fillna(0)

# Save the pre-processed dataset to a new file
df.to_csv("pre-processed_data.csv", index=False)

# Split the dataset into training and testing sets
# This is necessary to evaluate the performance of your model
df = pd.read_csv("pre-processed_data.csv")


# use plt to plot scatter plot of the data
plt.scatter(df['Year/Ports'],
            df['Petroleum Oil Lubricant(POL) & POL PRODUCTS'])
plt.show()


# Perform preliminary analysis on the dataset
# This may include generating summary statistics, visualizing the data,
# and identifying trends and patterns

# If using machine learning, choose an appropriate algorithm and train a model
# on your dataset

# Evaluate the performance of your model and make predictions
# on new data

# Write a comprehensive report on your findings and conclusions
# Be sure to include necessary plots, charts, and graphs to support your analysis
