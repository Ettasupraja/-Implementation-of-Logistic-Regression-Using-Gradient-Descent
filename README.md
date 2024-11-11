# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: ETTA SUPRAJA
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

Output:

![image](https://github.com/user-attachments/assets/5f256495-a58c-44a3-9dd2-f1ab1d82bf2b)

```
dataset.head()
```

Output:

![image](https://github.com/user-attachments/assets/1cf7e8fe-ce3d-474c-b424-346a5082ce2d)

```

```
dataset.tail()
```

Output:

![image](https://github.com/user-attachments/assets/fa73cd99-b205-4dc1-bb1f-f6e9f612b26a)

```
dataset.info()
```

Output:

![image](https://github.com/user-attachments/assets/85224ece-6115-46de-9c13-b70983a63d27)

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
Output:
![image](https://github.com/user-attachments/assets/aa9b28f9-9029-4fc9-ba79-461fd71da374)

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

Output:

![image](https://github.com/user-attachments/assets/999da40a-177b-4793-bd09-0614b108bdae)

```
dataset.info()
```
Output:
![image](https://github.com/user-attachments/assets/494d13ba-f485-454c-a21c-02a528c0f40a)

```
dataset.head()
```

Output:
![image](https://github.com/user-attachments/assets/21a79a3f-6c20-4ff3-857d-87876ff75e6b)

```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
X
Y
```

Output:
![image](https://github.com/user-attachments/assets/9e30eef3-5660-4fb5-b5ce-b2180e20190f)

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
Output:
![image](https://github.com/user-attachments/assets/f56f503f-60f1-4b69-bb97-51e195c91956)

```
print(y_pred)
```
Output:
![image](https://github.com/user-attachments/assets/35695902-40f2-4afd-b675-732dda03c01b)

```
print(Y)
```

Output:
![image](https://github.com/user-attachments/assets/c541fddd-7226-40f3-a8b0-be12cae2e0e4)

```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

Output:
![image](https://github.com/user-attachments/assets/9757c71f-c9a7-4753-a4d2-d48e463c193d)

```
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
OUtput:
![image](https://github.com/user-attachments/assets/0de52282-553b-42f6-b311-64611a441c14)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

