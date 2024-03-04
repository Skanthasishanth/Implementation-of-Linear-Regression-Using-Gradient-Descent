# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of 50 Startups using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import numpy, pandas, and StandardScaler from sklearn.preprocessing.
2. Read '50_Startups.csv' into a DataFrame (data) using pd.read_csv().
3. Extract features (X) and target variable (y) from the DataFrame.
4. Convert features to a numpy array (x1) and target variable to a numpy array (y).
5. Scale the features using StandardScaler() and define linear_regression(X1, y) function for linear regression.
6. Initialize theta as a zero vector.
7. Implement gradient descent to update theta.
8. Call linear_regression function with scaled features (x1_scaled) and target variable (y).
9. Prepare new data for prediction by scaling and reshaping.
10. Print the predicted value.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: S Kantha Sishanth 
RegisterNumber: 212222100020 
```
```py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):

# Add a column of ones to X for the intercept term 

X = np.c_[np.ones(len(X1)), X1]

# Initialize theta with zeros 

theta = np.zeros(X.shape[1]).reshape(-1,1)

# Perform gradient descent

for _ in range(num_iters):

# Calculate predictions 

predictions = (X).dot(theta).reshape(-1, 1)

# Calculate errors

errors = (predictions - y).reshape(-1,1)

# Update theta using gradient descent 

theta -= learning_rate * (1/ len(X1)) * X.T.dot(errors)
return theta

data = pd.read_csv('50_Startups.csv',header=None)

# Assuming the last column is your target variable 'y' and the preceding columns are your features 'X'

X = (data.iloc[1:, :-2].values)
X1=X.astype(float)

scaler = StandardScaler()

y = (data.iloc[1:,-1].values).reshape(-1,1)

X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)

# Learn model parameters

theta = linear_regression(X1_Scaled, Y1_Scaled)

# Predict target value for a new data point

new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot (np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre-scaler.inverse_transform(prediction)

print(f"Predicted value: {pre}")
```

## Output:



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
