# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:13:35 2020

@author: NARAYANA REDDY DATA SCIENTIST
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTING THE DATASET

dataset=pd.read_csv('Churn_Modelling.csv')

# DIVIDE THE DATASET INTO X AND Y
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

# ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1=LabelEncoder()
x[:,1]=labelencoder_x_1.fit_transform(x[:,1])
labelencoder_x_2=LabelEncoder()
x[:,2]=labelencoder_x_2.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
x=x[:,1:]

# SPLITTING THE DATASET INTO TRAIN AND TEST DATASET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# FITTING THE XGBOOST MODEL TO DATASET
from xgboost import XGBClassifier
classifier=XGBClassifier()

# PREDICT THE TEST  SET RESULTS 

y_predict=classifier.predict(x_test)

# CONFUSION MATRIX FOR MODEL PERFORMAMNCE

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)

