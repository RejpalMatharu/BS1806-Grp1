# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 16:30:49 2017

@author: Rejpal
"""

# loading libraries
import pandas as pd

# define column names
#names = ['fixed_acidity', 'volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulphur_dioxide','total_sulphur_dioxide','density','ph','sulphates','alcohol','quality']

#pandas load table
df = pd.read_table('winequality-red.csv', sep='\;')
df.head()

#add good quality wine
import numpy as np
l=[]

for number in df.ix[:,11]:
    if number >= 6:
        l.append(1)
    else:
        l.append(0)
        
df['good_wine'] = l

#split the dataframe into training and test data sets
from sklearn.utils import shuffle
df = shuffle(df)

training_df = df.iloc[0:800:,0:13]
test_df = df.iloc[800:1600:,0:13]

#normalise the training data set

z_training_df=(training_df-training_df.mean())/training_df.std()
z_test_df=(test_df-test_df.mean())/test_df.std()

#split into x train and y train, x test and y train
x_train = np.array(z_training_df.ix[:, 0:12]) 
y_train = np.array(training_df.ix[:, 12])

x_test = np.array(z_test_df.ix[:, 0:12]) 
y_test = np.array(test_df.ix[:, 12])

##K nearest neighbour model, k = 3
# loading library
from sklearn.neighbors import KNeighborsClassifier

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(x_train, y_train)

# predict the response on test data
pred = knn.predict(x_test)

# evaluate accuracy of test data against known y test values
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))

#k nearest neighbour for 1 to 500
# creating odd list of K for KNN
neighbors = list(range(1,500))

# empty list that will hold cross validation scores
cv_scores = []

# perform 5-fold cross validation
from sklearn.model_selection import cross_val_score

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
import matplotlib.pyplot as plt
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

##K nearest neighbour model
# loading library
from sklearn.neighbors import KNeighborsClassifier

# instantiate learning model
knn = KNeighborsClassifier(n_neighbors=optimal_k)

# fitting the model
knn.fit(x_test, y_test)

# predict the response on test data
pred = knn.predict(x_test)

# evaluate accuracy of test data against known y test values
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))

#confusion matrix
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_test, pred)
c_matrix

#total error rate
EMR_TER = (c_matrix[0][1]+c_matrix[1][0])/(c_matrix.sum())
EMR_TER

Accuracy = 1 - EMR_TER
Accuracy