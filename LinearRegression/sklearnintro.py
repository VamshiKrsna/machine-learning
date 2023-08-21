import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Advertising.csv")
print(df.head(5))

# sns.pairplot(df)

# Splitting train and test sets :

X = df.drop('sales',axis = 1)
print(X.head(5))

y = df['sales']
print(y.head(5))
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
from sklearn.model_selection import train_test_split
# print(help(train_test_split))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
# test_size : this gives the percentage of data to be spared to test datasets.
# Mostly 70 % data is used for training
# 30% is used for testing

print(X_train)
print(X_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LinearRegression

# print(help(LinearRegression))
model = LinearRegression()
print(model.fit(X_train,y_train))
print(model.predict(X_test))
XtPred = model.predict(X_test)

# sns.distplot(df['sales'],kde = True)
print("\n")
# sns.heatmap(df.corr())
print(model.intercept_)
print("\n")
print(model.coef_)

final_model = LinearRegression()
final_model.fit(X,y)
print(final_model.coef_)

print("\n")
# coef_ gives the deviation or change in each column when other two are kept constant.
# Here :
# [ 0.04576465  0.18853002 -0.00103749]
#    ^            ^          ^
#   TV          radio       newspaper    coeff's.
# newspaper's coeff generally means that there is no integral change in sales for any investment in newspaper sector.
# So, Model weighs a value near to 0 for newspaper column as it is nearly useless.
# For Better understanding :

fig, axes = plt.subplots(nrows= 1 ,ncols= 3, figsize = (16,6))

axes[0].plot(df['TV'],df['sales'],'o')
axes[0].set_title("TV Sales")
axes[0].set_ylabel("Sales")

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].set_title("radio Sales")
axes[1].set_ylabel("Sales")

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].set_title("newspaper Sales")
axes[2].set_ylabel("Sales")


# We can observe that there is no pattern of datapoints in newspaper vs sales plot.

# Lets predict yhat points i.e., y^

y_hat = final_model.predict(X)

axes[0].plot(df['TV'],df['sales'],'o',color = 'blue')
axes[0].plot(df['TV'],y_hat,'o',color = 'red')  # Red = Predictions, Blue = Base Datapoints.
axes[0].set_title("TV Sales")
axes[0].set_ylabel("Sales")


axes[1].plot(df['radio'],df['sales'],'o',color = 'blue')
axes[1].plot(df['radio'],y_hat,'o',color = 'red')
axes[1].set_title("radio Sales")
axes[1].set_ylabel("Sales")

axes[1].plot(df['newspaper'],df['sales'],'o',color = 'blue')
axes[2].plot(df['newspaper'],y_hat,'o',color = 'red')
axes[2].set_title("newspaper Sales")
axes[2].set_ylabel("Sales")

plt.tight_layout()

# Lets finalise our final model using joblib module

from joblib import dump,load

dump(final_model,'final_sales_model.joblib')
loaded_model = load('final_sales_model.joblib')

test1 = [[142,22,12]]
pred1 = loaded_model.predict(test1)
print(pred1)

print(df.columns)

from sklearn.metrics import mean_squared_error,mean_absolute_error

plt.show()
