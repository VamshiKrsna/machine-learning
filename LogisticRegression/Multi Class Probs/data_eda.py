# Multi Class Classification : Classifying more than 2 classes...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
print(df.head(5))

print(df.info)
print(df.describe())
print("\n")
print(df['species'].value_counts())

sns.countplot(data = df,x=df['species'])
fig = plt.figure(figsize=(12,8),dpi=100)
sns.scatterplot(x='petal_length',y = 'petal_width',data=df,hue = df['species'])
sns.pairplot(data=df,hue= 'species')
fig = plt.figure(figsize=(12,8),dpi = 100)
sns.heatmap(data = df.corr(),annot = True)

X = df.drop('species',axis=1)
y = df['species']

# print(y.value_counts())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y,
                    test_size=0.25, random_state=101)

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)



plt.show()