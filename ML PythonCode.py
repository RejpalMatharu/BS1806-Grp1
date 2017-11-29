# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:03:47 2017

@author: mingyang
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#Cleaning data to create a CSV readable by pandas
f=open("D:\Work\MSc Business Analytics\Machine Learning\Group Homework 1\winequality-red.csv", 'r')
a=f.read()
b=a.split('\n')
b[0] = b[0].replace('"', '')

for i in range(len(b)):
    b[i] = b[i].split(';')
    
    
#Creating pandas dataframe and converting type to numeric
wines = pd.DataFrame(b[1:], columns = b[0])
for col in wines:
    wines[col]=pd.to_numeric(wines[col], errors='raise')

f.close()
del(a, b, i, f)


#Creating new column for 'goodwine'
wines['good_wine'] = np.where(wines['quality']>=6, 1, 0)


#Removing null observation, splitting data into training and validation sets
wines = wines.drop(1599, axis=0)
winetrain, winetest = train_test_split(wines, test_size=0.5, random_state=123)


#Standardising all values in the dataframe, except for 'good_wine'
scaler = StandardScaler()
nwinetrain = pd.DataFrame(scaler.fit_transform(winetrain), index = winetrain.index, columns = winetrain.columns)
nwinetrain['good_wine'] = winetrain['good_wine'][:]


#Running 5-fold Cross Validation on all K's from 1 to 501. Recording results of each k in a dictionary.
num = 1
resultdict = {}
while num < 502:
    knn=KNeighborsClassifier(n_neighbors=num)
    xvres = cross_val_score(knn, X=nwinetrain.loc[:,'fixed acidity':'alcohol'], y=nwinetrain['good_wine'], cv=5, scoring='accuracy')
    resultdict[num] = xvres.mean()
    num+=5


#Saving best k result:
bestk = max(resultdict, key=resultdict.get)

knn=KNeighborsClassifier(n_neighbors=bestk)
knn.fit(winetrain.loc[:,'fixed acidity':'alcohol'], winetrain['good_wine'])
predictions = knn.predict(winetest.loc[:,'fixed acidity':'alcohol'])

pred1 = pd.DataFrame()
pred1['actual'] = winetest['good_wine'][:]
pred1['pred'] = predictions
    
confusion_matrix(y_true=pred1['actual'], y_pred=pred1['pred'])
