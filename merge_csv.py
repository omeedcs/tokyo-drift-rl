
import numpy as np
import pandas as pd

data_one = pd.read_csv("./bag-files-and-data/processed-data/1.csv")
data_two = pd.read_csv("./bag-files-and-data/processed-data/2.csv")
data_three = pd.read_csv("./bag-files-and-data/processed-data/3.csv")
data_four = pd.read_csv("./bag-files-and-data/processed-data/4.csv")
data_five = pd.read_csv("./bag-files-and-data/processed-data/5.csv")
data_six = pd.read_csv("./bag-files-and-data/processed-data/6.csv")
data_seven = pd.read_csv("./bag-files-and-data/processed-data/7.csv")
data_eight = pd.read_csv("./bag-files-and-data/processed-data/8.csv")

# rewrite pd concat
data = pd.concat([data_one, data_two, data_three, data_four, data_five, data_six, data_seven, data_eight], axis = 0) 
data = data.reset_index(drop = True)
data = data.drop(columns=["Unnamed: 0"])
data = data.sample(frac=1).reset_index(drop=True)
data.to_csv("./dataset/ikddata2.csv")
data = pd.read_csv("./dataset/ikddata2.csv")
def calculate_curvature(row):
    joystick = eval(row['joystick'])
    velocity = joystick[0]
    commanded_av = joystick[1]
    if velocity == 0.0:
        curvature = 0.0
    else:
        curvature = commanded_av / velocity
    return curvature

# apply the function to the dataframe and create a new column
data['curvature'] = data.apply(calculate_curvature, axis=1)

# save the updated data as CSV
data.to_csv("./dataset/ikddata2.csv")

# load the original data from CSV
data = pd.read_csv("./dataset/ikddata2.csv")

# drop rows where curvature is zero
data = data[data['curvature'] != 0]

# reset the index after dropping rows
data.reset_index(drop=True, inplace=True)

# save the updated data as CSV
data.to_csv("./dataset/ikddata2.csv")

# load the updated, cleaned data from CSV
data = pd.read_csv("./dataset/ikddata2.csv")

# drop the curvature column
data = data.drop('curvature', axis=1)

# save the updated data as CSV
data.to_csv("./dataset/ikddata2.csv")

# load the updated, cleaned data from CSV
data = pd.read_csv("./dataset/ikddata2.csv")
# drop the unnamed columns
data = data.drop(columns=["Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0"])

# drop the unnamed columns
data = data.drop(columns=["Unnamed: 0.3"])

# save merged csv
data.to_csv("./dataset/ikddata2.csv")
