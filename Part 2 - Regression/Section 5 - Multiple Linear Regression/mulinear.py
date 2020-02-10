#MULTIPLE LINEAR REGRESSION
#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets and making matrices
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc [ : , :-1].values
y = dataset.iloc [ : ,4 ].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#Avoiding Dummy Variable Trap
x = x[: , 1:]



# Splitting the datasets into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size= 0.2, random_state = 0 )

#feature scaling
"""from  sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

#fitting multiple linear regression to the test set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test results
y_pred = regressor.predict(x_test)

#building the optimal model for backward elimination
import statsmodels.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
x_opt = x[ : , [0,1,2,3,4,5,]]
regresson_OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor_OLS.summary()




