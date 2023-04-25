# Simple script to extract only joystick data from a CSV file
import pandas as pd

# Read the CSV file
data = pd.read_csv('dataset/ikddata2.csv')
data = data.drop(columns=["executed"])

# Extract the 'joystick' column
joystick = data['joystick']

# Write the joystick values to a text file
with open('output.txt', 'w') as file:
    for value in joystick:
        file.write(value + '\n')