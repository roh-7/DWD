import csv
import pandas as pd
import numpy as np
from gapminder import gapminder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

processed_data = pd.read_csv(
	'/Users/rohitramaswamy/Desktop/BSE/Sem 1/DealingWithData/git/DWD/Data/bank/bank-full.csv',sep=';')

# print(processed_data.info())

# print(processed_data['job'].unique().tolist())

print(processed_data["job"].value_counts())

# print("data length :: ", len(processed_data))
# print("no null data :: ", processed_data.isnull().sum())
# print(processed_data.head())
#
# processed_data.info()


# feature_col_x = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
#                  'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
#                  'cons.conf.idx', 'euribor3m', 'nr.employed']
# feature_col_y = ['y']

# X = processed_data.iloc[0:, 3:15]
# Y = processed_data.iloc[0:, -1]
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
#
# cal_entropy = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=3, min_samples_leaf=5)
# cal_entropy.fit(X_train, Y_train)
