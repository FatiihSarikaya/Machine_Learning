import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


import numpy as np

data = pd.read_csv('data.csv')
print(data.head())
print(data.info())
data.drop(['Unnamed: 32','id'],axis=1,inplace=True)

nulls = sns.heatmap(data.isnull())
plt.show()

data['diagnosis'] = [ 1 if value == 'M' else 0 for value in data['diagnosis']]
data['diagnosis'] = data['diagnosis'].astype('category',copy=False)
data['diagnosis'].value_counts().plot(kind = 'bar')
plt.show()
#divide into target variable and predictors
y = data['diagnosis'] #target variable
x = data.drop(['diagnosis'],axis=1)
print(y)
print(x)
#normalization
#create a scaler object
scaler = StandardScaler()

#fit the scaler to the data and transform the data

x_scaled = scaler.fit_transform(x)
print(x_scaled)

#split the data so gonna test and train the data

x_train , x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.30, random_state=42)

#train the model
#create the lr model

lr = LogisticRegression()

#train the model on the training data

lr.fit(x_train, y_train)

#predict the target variable based on test data

y_pred = lr.predict(x_test)
print(y_pred)

#evaluation of the model

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:  {accuracy: .2f}")#0.98

print(classification_report(y_test,y_pred))











