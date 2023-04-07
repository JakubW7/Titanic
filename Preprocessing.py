import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def proces_data(given_data):
    label = 0
    given_data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket', 'Age'], axis=1, inplace=True)
    if 'Survived' in given_data.columns:
        label = given_data['Survived']
        given_data.drop(['Survived'], axis=1, inplace=True)
    given_data = pd.concat([pd.get_dummies(given_data[['Embarked', 'Sex']], drop_first=False),
                            given_data.drop(['Embarked', 'Sex'], axis=1)], axis=1)
    given_data.drop(['Embarked_Q', 'Embarked_C', 'Sex_male'], axis=1, inplace=True)
    return given_data, label


def transform_data(train_data, test_data):
    scaler = StandardScaler()
    scaler.fit_transform(train_data)
    scaler.transform(test_data)
    imputer = SimpleImputer(strategy='median')
    train_data[:] = imputer.fit_transform(train_data)
    test_data[:] = imputer.transform(test_data)
    return train_data, test_data
