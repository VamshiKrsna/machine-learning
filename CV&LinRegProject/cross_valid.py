import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Advertising.csv")
# print(df.head(5))

X = df.drop('sales',axis=1)
y = df['sales']
# print(X)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# T_T_S
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state= 101)

# Scaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Model Selection
from sklearn.linear_model import Ridge

model = Ridge(alpha=100)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import  mean_squared_error
print(mean_squared_error(y_pred,y_test))

model = Ridge(alpha=10)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(mean_squared_error(y_pred,y_test))

model = Ridge(alpha=1)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(mean_squared_error(y_pred,y_test))
