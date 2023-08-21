import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Advertising.csv")

from sklearn.model_selection import train_test_split

X = df.drop('sales',axis = 1)
y = df['sales']

X_train, X_other, y_train, y_other = train_test_split(X,y,test_size=0.3,random_state= 101)
X_eval, X_test, y_eval, y_test = train_test_split(X_other,y_other,test_size= 0.5,random_state=101)
# 50% of 30 % other data
# meaning 15% of overall total data.

# eval = validation data set.

print(len(df))
print("\n")
print(len(X_train))
print(len(X_test))
print(len(X_eval))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_eval = scaler.transform(X_eval)

from sklearn.linear_model import  Ridge

model = Ridge(alpha= 100)
model.fit(X_train,y_train)

y_eval_pred = model.predict(X_eval)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_eval,y_eval_pred))

model_2 = Ridge(alpha=0.1)
model_2.fit(X_train,y_train)

y_eval_pred2 = model_2.predict(X_eval)
print(mean_squared_error(y_eval,y_eval_pred2))

y_final_test_preds = model_2.predict(X_test)  # y_test predictions from model2

print(mean_squared_error(y_test,y_final_test_preds)) # Good score...


