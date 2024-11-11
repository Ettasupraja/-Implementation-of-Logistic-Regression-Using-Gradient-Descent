# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step-1:Start
Step-2:Import the necessary python packages
Step-3:Read the dataset.
Step-4:Define X and Y array.
Step-5:Define a function for costFunction,cost and gradient.
Step-6:Define a function to plot the decision boundary and predict the Regression value.
Step-7: End
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Etta Supraja
RegisterNumber:212223220022
*/
```
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score


```
```
dataset=pd.read_csv('Placement.csv')
print(dataset)


```
### Output:
![image](https://github.com/user-attachments/assets/acdf306f-06cb-4a2b-98ef-ba70a0487cc4)
```
dataset.head()


```
### Output:
![image](https://github.com/user-attachments/assets/a815db95-643c-4f10-8b0a-3041e484f42d)
```
dataset.tail()


```
### Output:
![image](https://github.com/user-attachments/assets/8e7490bd-0fb2-4368-871c-770c336f98ba)
```
dataset.info()


```
### Output:
![image](https://github.com/user-attachments/assets/19e73724-93b1-4c19-b947-4d19925f42d7)
```
dataset.drop('sl_no',axis=1,inplace=True)


```
```
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes


```
### Output:
![image](https://github.com/user-attachments/assets/7bd6179a-135f-4626-82fb-1de700f2b8e0)
```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset


```
### Output:
![image](https://github.com/user-attachments/assets/46e63937-2a28-4d14-a341-3c0fbe65f47e)
```
dataset.info()


```
### Output:
![image](https://github.com/user-attachments/assets/dfdbba5c-e993-4c05-a06d-2c7a2140b017)
```
dataset.head()


```
### Output:
![image](https://github.com/user-attachments/assets/9bc09834-e043-435d-b3a6-2e0d6f71fd50)
```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
X
Y


```
### Output:
![image](https://github.com/user-attachments/assets/367f34bb-294a-4996-8c69-a09980193c6b)
![image](https://github.com/user-attachments/assets/a1776a01-f646-41de-be92-3617743db71c)
```
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,X,Y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(x.dot(theta))
    gradient=x.T.dot(h-y)/m
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01, num_iterations=1000)
```
```
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,X)


```
```
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)


```
### Output:
![image](https://github.com/user-attachments/assets/69f7bb24-d488-4a22-961f-be46e9499808)
```
print(y_pred)


```
### Output:
![image](https://github.com/user-attachments/assets/d03776ee-dcbf-45e3-be6b-f690ec6944a2)
```
print(Y)

```
### Output:
![image](https://github.com/user-attachments/assets/9a956ce3-0ac8-432a-9b09-5a18d5d8d37e)
```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)


```
### Output:
![image](https://github.com/user-attachments/assets/620e69af-82b7-4da0-8d54-ccb10843ca15)
```
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)


```
### Output:
![image](https://github.com/user-attachments/assets/c96429e4-ab59-41cb-b2d3-4a37f3a2d38e)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


