# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 22:54:14 2019

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\HP\\Desktop\\u_datasets\\Logistic_Regression\\Social_Network_Ads.csv")
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(x)
x=scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.20,random_state=0)


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',max_depth=6,random_state=0)
classifier.fit(x_train,y_train)
classifier.score(x,y)
classifier.score(x_train,y_train)
classifier.score(x_test,y_test)

from sklearn.model_selection import cross_val_score,KFold
kfold=KFold(n_splits=7,random_state=0)
score=cross_val_score(classifier,x,y,cv=kfold,scoring='accuracy')
score.mean()
print('score:',score.mean())

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus

features=list(dataset.columns[2:4])
features

dot_data=StringIO()
export_graphviz(classifier,out_file=dot_data,feature_names=features,filled=True,rounded=True,class_names=['0','1'])
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())

import os
os.environ["PATH"] += os.pathsep + "E:/graphviz-2.38/release/bin"

graph.write_png("ads.png")
Image(graph.create_png())
