# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:33:14 2017

@author: Yiting
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('winequality-red.csv', sep = ';')

def gw(x):
    if x >= 6:
        return 1
    else:
        return 0
    
df['good wine'] = df.quality.apply(gw)

from sklearn.utils import shuffle

df = shuffle(df)

df_features = df.iloc[:,0:11]
df_labels = df.iloc[:, 12]

avg = df_features.mean(axis = 0)
std = df_features.std(axis = 0)

df_features = (df_features - avg) / std

train_features = df_features[0: 800]
train_labels = df_labels[0: 800]

test_features = df_features[800: ]
test_labels = df_labels[800: ]

from sklearn.cross_validation import cross_val_score

accu = {}
k = 1
while k <= 500:
    knn = KNeighborsClassifier(n_neighbors = k)
    score = cross_val_score(knn, X = train_features,
                            y = train_labels, cv = 5,scoring='accuracy')
    accu[k] = score.mean()
    k = k + 5

max_v = 0
for k,v in accu.items():
    if v > max_v:
        max_v = v
        max_k = k
print(max_v, max_k)
# k = 112

knn_best = KNeighborsClassifier(n_neighbors = max_k)
knn_best.fit(train_features, train_labels)
accur_test = knn_best.score(test_features, test_labels)
general_error = 1 - accur_test
general_error

from sklearn.metrics import confusion_matrix

test_pred = knn_best.predict(test_features)
confusion_matrix(test_labels, test_pred)
