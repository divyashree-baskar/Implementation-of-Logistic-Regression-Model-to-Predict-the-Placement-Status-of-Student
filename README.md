# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries like `pandas` for data manipulation and `numpy` for numerical operations.
2. Load the CSV file (`Placement_Data.csv`) into a pandas DataFrame and preview the first 5 rows using `df.head()`.
3. Make a copy of the original DataFrame (`df`) to preserve the original data.
4. Drop the columns `'sl_no'` (serial number) and `'salary'` because they are not required for modeling.
5. `.isnull().sum()` returns the count of missing values for each column.
6. `.duplicated().sum()` returns the count of duplicate rows in the dataset.
7. The `LabelEncoder` from `sklearn` is used to transform string labels into numeric labels.
8. This is done for columns `'gender'`, `'ssc_b'`, `'hsc_b'`, `'hsc_s'`, `'degree_t'`, `'workex'`, `'specialisation'`, and `'status'` where each unique category is assigned a numeric value.
9. `'x'` is created by selecting all columns except for the last column (`'status'`), which is the target variable.
10. `'y'` is the `'status'` column, which represents whether the student got placed or not (binary classification).
11. The data is split using `train_test_split` from `sklearn`.
12. `test_size=0.2` means that 20% of the data will be used for testing, and 80% will be used for training.
13. `random_state=0` ensures that the split is reproducible.
14. `LogisticRegression(solver="liblinear")` creates a logistic regression classifier using the `'liblinear'` solver (good for smaller datasets).
15. `lr.fit(x_train, y_train)` trains the model on the training data.
16. Use the trained logistic regression model (`lr`) to predict the target values (`'status'`) for the test set (`x_test`).
17. `accuracy_score` compares the predicted values (`y_pred`) with the true values (`y_test`) and calculates the accuracy of the model.
18. `confusion_matrix(x_test, y_pred)` outputs the confusion matrix based on the true values and predicted values.
19. `classification_report(y_test, y_pred)` provides metrics such as precision, recall, F1-score, and support for each class (in this case, whether a student got placed or not).
20. The input should match the features used in the model (numeric values representing different attributes of the student).

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DIVYASHREE B
RegisterNumber: 212224040081
import pandas as pd
import numpy as np
df=pd.read_csv('Placement_Data.csv')
print("Name: SWETHA S\nReg.no: 212224040344)
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis = 1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1['gender']=le.fit_transform(df1['gender'])
df1['ssc_b']=le.fit_transform(df1['ssc_b'])
df1['hsc_b']=le.fit_transform(df1['hsc_b'])
df1['hsc_s']=le.fit_transform(df1['hsc_s'])
df1['degree_t']=le.fit_transform(df1['degree_t'])
df1['workex']=le.fit_transform(df1['workex'])
df1['specialisation']=le.fit_transform(df1['specialisation'])
df1['status']=le.fit_transform(df1['status'])
df1

x=df1.iloc[:,:-1]
x

y=df1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:

<img width="1165" height="283" alt="Screenshot 2025-09-24 000007" src="https://github.com/user-attachments/assets/e393bb80-270e-4d1a-96f5-618a003089e2" />

<img width="1025" height="202" alt="image" src="https://github.com/user-attachments/assets/a6ac43dd-5577-4c3e-b1c6-a3e3fd73b8b4" />

<img width="213" height="316" alt="image" src="https://github.com/user-attachments/assets/0a9df4b8-047b-4bf1-a22e-93a7938c7e6c" />  <img width="34" height="30" alt="image" src="https://github.com/user-attachments/assets/333626de-d2b5-48ca-907a-db73059eb7d5" />

<img width="935" height="427" alt="image" src="https://github.com/user-attachments/assets/e1b0a8e1-2344-4084-a953-ceacc2c98565" />

<img width="880" height="435" alt="image" src="https://github.com/user-attachments/assets/4fd4c14d-2c86-44b4-a5e5-133df3c85394" />

<img width="388" height="263" alt="image" src="https://github.com/user-attachments/assets/d02cd38c-ffc1-4093-96b2-5d1652910768" />

<img width="702" height="61" alt="image" src="https://github.com/user-attachments/assets/8e05b896-200d-4c91-8c2d-bbb298fcc9e5" />   <img width="184" height="34" alt="image" src="https://github.com/user-attachments/assets/76af680b-25c7-4484-81cf-a913b08d4acd" />

<img width="298" height="67" alt="image" src="https://github.com/user-attachments/assets/0144cf85-80e2-47fa-9140-dbca6884004a" />  <img width="529" height="186" alt="image" src="https://github.com/user-attachments/assets/49686ee3-3199-459c-bc73-a00eac0dfc50" />  <img width="106" height="31" alt="image" src="https://github.com/user-attachments/assets/f9049cc7-8a57-44a5-8a41-5a5ec1b789d0" />













## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
