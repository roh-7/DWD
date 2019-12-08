from sklearn.ensemble import RandomForestClassifier

#Create the model with 100 trees
random_forest_model = RandomForestClassifier(n_estimators = 100, bootstrap = True, max_features = 'sqrt')
# Fit on training data
random_forest_model.fit(X_train, Y_train)

#
Y_pred = random_forest_model.predict(X_test)