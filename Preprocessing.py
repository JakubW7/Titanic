import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def proces_data(data):
    """Drop unimportant columns from data, one hot encode particular columns, return features and label"""
    label = 0
    data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket', 'Age'], axis=1, inplace=True)
    if 'Survived' in data.columns:
        label = data['Survived']
        data.drop(['Survived'], axis=1, inplace=True)
    data = pd.concat([pd.get_dummies(data[['Embarked', 'Sex']], drop_first=False),
                            data.drop(['Embarked', 'Sex'], axis=1)], axis=1)
    data.drop(['Embarked_Q', 'Embarked_C', 'Sex_male'], axis=1, inplace=True)
    return data, label


def transform_data(train_data, test_data):
    """Scale and impute train and test data without data contamination"""
    scaler = StandardScaler()
    scaler.fit_transform(train_data)
    scaler.transform(test_data)
    imputer = SimpleImputer(strategy='median')
    train_data[:] = imputer.fit_transform(train_data)
    test_data[:] = imputer.transform(test_data)
    return train_data, test_data
