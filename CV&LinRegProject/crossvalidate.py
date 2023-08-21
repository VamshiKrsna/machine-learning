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

from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge

model1 = Ridge(alpha=100)

scores = cross_validate(model1,X_train,y_train,scoring=['neg_mean_squared_error',
                                                        'neg_mean_absolute_error'],
                        cv = 10)

# print(scores)  # Ugly...
score_df = pd.DataFrame(scores)
# print(score_df)
for col in score_df:
    print(score_df[col])

print(score_df.mean())
print(abs(score_df.mean()))

from sklearn.metrics import mean_squared_error,mean_absolute_error


model1.fit(X_train,y_train)
y_fake_preds = model1.predict(X_test)

print("\n")

print(mean_squared_error(y_test,y_fake_preds))
print(mean_absolute_error(y_test,y_fake_preds))


model_final = Ridge(alpha=1)

model_final.fit(X_train,y_train)

y_final_preds = model_final.predict(X_test)



print("\n")

print(mean_squared_error(y_test,y_final_preds))
print(mean_absolute_error(y_test,y_final_preds))
