from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import joblib
data= pd.read_csv('TitanicDataset.csv')

data.drop('zero', axis=1, inplace=True)
print(data.head())

print('Values of sex column')
print(pd.get_dummies(data['Sex']))
Sex= pd.get_dummies(data['Sex'], drop_first=True)
print(Sex.head())

print('Values of pass column after removing one field')
Pclass= pd.get_dummies(data['Pclass'], drop_first=True)
print(Pclass.head())

print('Data after Concatinating all')
data= pd.concat([data, Sex, Pclass], axis=1)
print(data.head())
print('Data after droping useless columns')
data.drop(['Passengerid', 'sibsp', 'Parch', 'Embarked', 'Sex', 'Pclass'], axis=1, inplace=True)
print(data.head())



X= data.drop('Survived', axis=1)
y= data['Survived']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=1)

model= LogisticRegression()

model.fit(X_train, y_train)

pred= model.predict(X_test)

print('Classification report is : ', classification_report(y_test, pred))

print('Confusion matrix is : ', confusion_matrix(y_test, pred))

print('Accuracy score is : ', accuracy_score(y_test, pred))

filename = 'TitanicModel'
joblib.dump(model, filename)

print("Training completed TitanicModel generated")