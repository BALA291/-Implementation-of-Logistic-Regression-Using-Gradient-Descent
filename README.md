# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: BALAMURUGAN B
RegisterNumber:  212222230016
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("Placement_Data.csv")
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))
def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Y_PRED:",y_pred)
print("Y:",Y)
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

```


## Output:
![m5i](https://github.com/BALA291/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120717501/d1fb2c58-a2ea-47ce-85af-b49fb45860ca)
![m5ii](https://github.com/BALA291/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120717501/ff2650f0-ae15-475b-a3ed-a2ba65852969)
![m5iii](https://github.com/BALA291/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120717501/8a4bbbd5-3fa5-4a8e-9db4-12e1bdfa909c)
![m5iv](https://github.com/BALA291/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120717501/7ddd5870-5be8-4223-865e-ae325ff34aab)
![m5v](https://github.com/BALA291/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120717501/a84d94ec-5d73-4a6e-b5fe-bbd6874f2606)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

