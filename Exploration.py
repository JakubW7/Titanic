import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import Preprocessing

# Read and preprocess data
file = pd.read_csv('titanic/train.csv')
data, y = Preprocessing.proces_data(file)

# Show basic statistics derived from data
sns.pairplot(data)
plt.show()
print(data.describe())
print(data.var())
print(data.info())
print(data.corr())

# Find importance of all features
X_train, X_test, y_train, y_test = train_test_split(data, y, random_state=0)
X_train, X_test = Preprocessing.transform_data(X_train, X_test)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))
for i, feature in enumerate(model.feature_importances_):
    print(X_train.columns[i], feature)
