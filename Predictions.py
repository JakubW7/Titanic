import pandas as pd
import xgboost
from sklearn.ensemble import RandomForestClassifier
import Preprocessing

#Reading data
X_train = pd.read_csv('titanic/train.csv')
X_test = pd.read_csv('titanic/test.csv')
PassengerId = X_test['PassengerId']

#Preprocessing
X_train, y_train = Preprocessing.proces_data(X_train)
X_test, y_test = Preprocessing.proces_data(X_test)
X_train, X_test = Preprocessing.transform_data(X_train,X_test)

#Creating models
rf = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_leaf=3, n_estimators=100)
xgb = xgboost.XGBClassifier(alpha=0.0001, gamma=0.001, learning_rate=0.11, max_depth=2, n_estimators=50)

#Choosing model, fitting, preditctions
model = xgb
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
Final_data = pd.concat([PassengerId, pd.Series(y_pred, name='Survived')], axis=1)
Final_data.to_csv('out.csv', index=False)
