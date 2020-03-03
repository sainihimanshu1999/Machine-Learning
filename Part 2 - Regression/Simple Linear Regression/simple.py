#simple regression model
#importing Libraries
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets and making matrices
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc [ : , :-1].values
y = dataset.iloc [ : ,1 ].values


# Splitting the datasets into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size= 1/3, random_state = 0 )

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set results    
y_pred = regressor.predict(x_test)

#Visualising the training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()



#feature scaling
"""from  sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

