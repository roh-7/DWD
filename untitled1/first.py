import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

processed_data=pd.read_csv('D:\Books\MFT\Projects\dealing with data group 1\DATA\processed_data.csv', sep =';', header = 1)

#print("data length :: ", len(processed_data))
print("no null data :: ", processed_data.isnull().sum())
print(processed_data.head())

#feature_col = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

X = processed_data.iloc[1:, 0:19]
Y = processed_data.iloc[1:, -1]

X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

cal_entropy = DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
cal_entropy.fit(X_train, Y_train)
