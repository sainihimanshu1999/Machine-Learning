#importing Libraries
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets and making matrices
dataset = pd.read_csv('Data.csv')
x = dataset.iloc [ : , :-1].values
y = dataset.iloc [ : ,3 ].values


# Splitting the datasets into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size= 0.2, random_state = 0 )

#feature scaling
"""from  sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""
