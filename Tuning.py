import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# Importing and preprocessing data
data = pd.read_csv('titanic/train.csv')
X = data.drop(['Cabin', 'Ticket', 'Name', 'Age', 'PassengerId'], axis=1)
X = X.dropna()
data_dummies = pd.get_dummies(X[['Sex', 'Embarked']], drop_first=True)
X = pd.concat([X, data_dummies], axis=1)
X = X.drop(['Sex', 'Embarked'], axis=1)
y = X['Survived']
X = X.drop(['Survived'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Scaling both sets
scaler= preprocessing.StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate individual classifiers
lr = LogisticRegression(random_state=0, C=0.01, solver='newton-cg', penalty='l2')
knn = KNeighborsClassifier(n_neighbors=14)
dt = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_leaf=1, criterion='entropy')
rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1, min_samples_leaf=6, max_depth=4,
                            criterion='gini')
xb = xgb.XGBClassifier(alpha=0.0001,gamma=0.001,learning_rate=0.11,max_depth=2,n_estimators=50)

# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn,), ('Classification Tree', dt,), ('RF', rf,),
               ('XGB', xb)]

# Create hyperparameters dictionary
param_dic = {'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'],
                                     'solver': ['newton-cg', 'lbfgs', 'liblinear']},
             'K Nearest Neighbours': {'n_neighbors': np.arange(1, 20)},
             'Classification Tree': {'criterion': ['gini', 'entropy', 'log_loss'], 'max_depth': np.arange(1, 10),
                                     'min_samples_leaf': np.arange(1, 10)},
             'RF': {'criterion': ['gini'], 'max_depth': np.arange(3, 7), 'min_samples_leaf': np.arange(3, 7),
                    'n_estimators': [100]},
             'XGB': {'learning_rate': np.arange(0.01, 0.3, 0.1), 'max_depth': np.arange(1, 3, 1), 'n_estimators': [50],
                     'alpha': [0.0001, 0.001, 0.01], 'gamma': [0.001, 0.01, 0.1]}}

# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    grid = GridSearchCV(estimator=clf,
                        param_grid=param_dic[clf_name], cv=KFold(n_splits=5), scoring='accuracy')
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    y_pred_train = grid.predict(X_train)
    print('best params', clf_name, grid.best_params_)
    print('best score', clf_name, grid.best_score_)
    print('Test score {:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))
    print('Train score', clf_name, accuracy_score(y_train, y_pred_train))

# Create, fit and get score of VotingClassifier
vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)
y_pred_train = vc.predict(X_train)
print('Voting Classifier: {}'.format(accuracy_score(y_test, y_pred)))
print('Train score ', accuracy_score(y_train, y_pred_train))
print('Voting clas', np.mean(cross_val_score(vc, X_train, y_train, cv=5)))
