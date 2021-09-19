# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 20:35:47 2021

@author: M Shoaib
"""


import numpy as np
import sklearn as sk 
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression  
from sklearn import metrics
url="http://bit.ly/w-data"
#read data
df=pd.read_csv(url)
#check for null values if exits drop them
df=df.dropna()
X=df[['Hours']]
y=df[['Scores']]
#print(df.head(10))
df.plot(x='Hours',y='Scores',style='x')
plt.title('Total Study Hours vs Scores in Perncentage')
plt.xlabel('No. of Hours Studied')
plt.ylabel('Percentage of Scores')
plt.show
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 23)
lg=LinearRegression()
print('Starting Training')
print('.')
print('..')
print('...')
lg.fit(X_train,y_train)
print('Training Completed :)')
line=lg.coef_*X + lg.intercept_
plt.scatter(X,y)
plt.plot(X,line,color='red',label='Linear Regression Line')
#printing actual results vs predicted results of the model
y_pred=lg.predict(X_test)
fl=y_pred.flatten()
# df1=pd.DataFrame({'Real Results':y_test,'Predicted Results':y_pred})
df1=pd.DataFrame()
df1['Predicted Results']=fl
y_true=y_test[['Scores']]
y_true=y_true.values.tolist()
flat_list = [item for sublist in y_true for item in sublist]
df1['True Results']=flat_list
print(df1)
#now our estimatng training and testing score
print('Testing Score is: ',lg.score(X_train,y_train))
print('Training Score is: ',lg.score(X_test,y_test))
# Testing with our question
print('Now testing with our Question: ')
hours =9.25
new_test=np.array([hours])
new_test=new_test.reshape(-1,1)
p=lg.predict(new_test)
print('No of hours: ',hours)
print('Predicted Score is: ',p[0])
#now evaluating
mad=metrics.mean_absolute_error(y_test,y_pred)
r2=metrics.r2_score(y_test,y_pred)
print('Mean Absolute Error is :',mad)
print('R-2 or the model accuracy is :',r2)

