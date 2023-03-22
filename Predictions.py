import pandas as pd
import xgboost
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('titanic/train.csv')
X = data.drop(['Cabin', 'Ticket', 'Name','Age','PassengerId'], axis=1)
X = X.dropna()
data_dummies = pd.get_dummies(X[['Sex', 'Embarked']], drop_first=True)
print(X.isna().sum())
X = pd.concat([X, data_dummies], axis=1)
X = X.drop(['Sex', 'Embarked'], axis=1)
y = X['Survived']
X_train = X.drop(['Survived'], axis=1)

X_test=pd.read_csv('titanic/test.csv')
PassengerId=X_test['PassengerId']
X_test =X_test.drop(['Cabin', 'Ticket', 'Name','Age','PassengerId'], axis=1)
data_dummies = pd.get_dummies(X_test[['Sex', 'Embarked']], drop_first=True)
print(X_test.isna().sum())
X_test = pd.concat([X_test, data_dummies], axis=1)
X_test = X_test.drop(['Sex', 'Embarked'], axis=1)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model=xgboost.XGBClassifier(alpha=0.0001,gamma=0.001,learning_rate=0.2,max_depth=2,n_estimators=50)
model.fit(X_train,y)
y_pred=model.predict(X_test)
print(y_pred)
Final_data=pd.concat([PassengerId,pd.Series(y_pred,name='Survived')],axis=1)
print(Final_data)
Final_data.to_csv('out.csv',index=False)






