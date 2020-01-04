import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

df = pd.read_csv("D:\Books\MFT\Projects\dealing with data group 1\DATA/bank-full.csv", sep=';')
pro_data = df

pro_data['job']=pd.factorize(pro_data.job)[0]
pro_data['marital']=pd.factorize(pro_data.marital)[0]

cleanup_education = {"education": {"primary": 1, "secondary": 2, "tertiary": 3, "unknown": 0}}
pro_data.replace(cleanup_education, inplace=True)

cleanup_boolean = {"default": {"yes": 1, "no": 0},
                   "housing": {"yes": 1, "no": 0},
                   "loan": {"yes": 1, "no": 0},
                   "y": {"yes":1, "no":0}
                   }
pro_data.replace(cleanup_boolean, inplace=True)

pro_data['contact']=pd.factorize(pro_data.contact)[0]
pro_data['month']=pd.factorize(pro_data.month)[0]

cleanup_poutcome = {"poutcome": {"success":1, "failure":2, "other":3, "unknown":4}}
pro_data.replace(cleanup_poutcome, inplace=True)

pro_data.to_csv('check1.csv')

X = pro_data.iloc[1:, 0:15]
Y = pro_data.iloc[1:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

cal_entropy = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=3, min_samples_leaf=5)
cal_entropy.fit(X_train, Y_train)
Y_pred = cal_entropy.predict(X_test)
print('Decision tree accuracy: ', metrics.accuracy_score(Y_test, Y_pred))
