import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Advertising.csv")

from sklearn.model_selection import train_test_split

X = df.drop('sales',axis = 1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

model = Ridge(alpha= 100)

scores = cross_val_score(model, X_train,y_train,scoring='neg_mean_squared_error',
                         cv = 5)
print(scores)
print(abs(scores.mean()))

model2 = Ridge(alpha=10)
scores2 = cross_val_score(model2, X_train,y_train,scoring='neg_mean_squared_error',
                         cv = 5)
print(scores2)
print(abs(scores2.mean()))

model3 = Ridge(alpha=1)
scores3 = cross_val_score(model3, X_train,y_train,scoring='neg_mean_squared_error',
                         cv = 5)
print(scores3)
print(abs(scores3.mean())) # better score.
# So, finalize model3...

# Finalising the model.

print("\n")

model3.fit(X_train,y_train)
y_final_test_pred = model3.predict(X_test)
print(mean_squared_error(y_test,y_final_test_pred))
