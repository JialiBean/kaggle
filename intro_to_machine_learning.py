# 2021/07/22

# Basic data exploration

import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Print summary statistics in next line
home_data.describe()

# What is the average lot size (rounded to nearest integer)?
avg_lot_size = 10517

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = 11

# 2021/07/23
# Your first machine learning model

# print the list of columns in the dataset to find the name of the prediction target
home_data.columns

y = home_data.SalePrice

# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Review data
# print description or statistics from X
print(X.describe())

# print the top few lines
print(X.head())

from sklearn.tree import DecisionTreeRegressor
#specify the model.
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X,y)

predictions = iowa_model.predict(X)
print(predictions)

print("Making predictions for the first values:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))
print("y first values are:")
print(y.head())