import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from constants import path

csv_path = path
processed_data = pd.read_csv(csv_path, sep=';')

processed_encoding = processed_data

# prints count of unique values
# print(processed_data["job"].value_counts())
cleanup_jobs = {"job": {"blue-collar": 1, "management": 2, "technician": 3, "admin.": 4, "services": 5, "retired": 6,
                        "self-employed": 7, "entrepreneur": 8, "unemployed": 9, "housemaid": 10, "student": 11,
                        "unknown": 12}}
processed_encoding.replace(cleanup_jobs, inplace=True)
# print(processed_jobs.head())

# print(processed_encoding['marital'].value_counts())
cleanup_marital = {"marital": {"married": 1, "single": 2, "divorced": 3}}
processed_encoding.replace(cleanup_marital, inplace=True)
# print(processed_encoding.head())

# print(processed_encoding["education"].value_counts())
cleanup_education = {"education": {"primary": 1, "secondary": 2, "tertiary": 3, "unknown": 4}}
processed_encoding.replace(cleanup_education, inplace=True)
# print(processed_encoding)

# for default, housing, loan
# print(processed_encoding["loan"].value_counts())
cleanup_boolean = {"default": {"yes": 1, "no": 0},
                   "housing": {"yes": 1, "no": 0},
                   "loan": {"yes": 1, "no": 0},
                   "y": {"yes":1, "no":0}
                   }
processed_encoding.replace(cleanup_boolean, inplace=True)
# print(processed_encoding.values[1:100, 3:9]

# print(processed_encoding["contact"].value_counts())
cleanup_contact = {"contact": {"cellular": 1, "telephone": 2, "unknown": 3}}
processed_encoding.replace(cleanup_contact, inplace=True)
# print(processed_encoding[['age']])

# print(processed_encoding["month"].value_counts())
cleanup_month = {"month": {"jan": 1, "feb":2, "mar":3, "apr":4, "may":5, "jun": 6, "jul":7, "aug":8, "sep":9,
                           "oct":10, "nov":11, "dec":12}}
processed_encoding.replace(cleanup_month, inplace=True)
# print(processed_encoding[['month']])

# print(processed_encoding["poutcome"].value_counts())
cleanup_poutcome = {"poutcome": {"success":1, "failure":2, "other":3, "unknown":4}}
processed_encoding.replace(cleanup_poutcome, inplace=True)
# print(processed_encoding[['poutcome']])

# print("data length :: ", len(processed_data))
# print("no null data :: ", processed_data.isnull().sum())
# print(processed_data.head())

# processed_data.info()

X = processed_encoding.iloc[1:, 0:15]
Y = processed_encoding.iloc[1:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

def decision_tree():
    cal_entropy = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=3, min_samples_leaf=5)
    cal_entropy.fit(X_train, Y_train)
    Y_pred = cal_entropy.predict(X_test)
    print('Decision tree accuracy: ', metrics.accuracy_score(Y_test, Y_pred))
    #  PRoc data: 0.8897736488977365

def random_forest():
    # Create the model with 100 trees
    random_forest_model = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
    # Fit on training data
    random_forest_model.fit(X_train, Y_train)
    Y_pred = random_forest_model.predict(X_test)
    print("Random forest accuracy: ",metrics.accuracy_score(Y_test,Y_pred))
    # 0.9023077490230775

def naive_bayes():
    model = GaussianNB()
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    print("Naive bayes accuracy: ", metrics.accuracy_score(Y_test, Y_pred))
    # 0.8607977586079776

def kmeans():
    # kmeans5 = KMeans(n_clusters=5)
    # y_kmeans5 = kmeans5.fit_predict(X_test)
    # print(y_kmeans5)
    # print(kmeans5.cluster_centers_)
    # Error = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i).fit(X_test)
    #     kmeans.fit(X_test)
    #     Error.append(kmeans.inertia_)
    # plt.plot(range(1, 11), Error)
    # plt.title('Elbow method')
    # plt.xlabel('No of clusters')
    # plt.ylabel('Error')
    # plt.show()
    print("-----------------------------")
    kmeans3 = KMeans(n_clusters=3)
    y_kmeans3 = kmeans3.fit_predict(X_test)
    print(y_kmeans3)
    print(kmeans3.cluster_centers_)
    plt.scatter(X_test[:,0], X_test[:,1],c=y_kmeans3,cmap='rainbow')





# decision_tree()
# random_forest()
# naive_bayes()
kmeans()
