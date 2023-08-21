import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Rizen3\\Desktop\\vamshi\\VamshiProgrammingStuff\\PythonDS&ML\\PythonPDJP2021\\DATA\\Advertising.csv")

print(df.head())
# print(df.describe())

X = df.drop('sales',axis = 1)
y = df['sales']

print(X.head())
print("\n")
print(y.head())

from sklearn.preprocessing import PolynomialFeatures

polynomial_converter = PolynomialFeatures(degree=2,include_bias=False)
# include_bias will add a to the power 0 terms to the model too.(if set True).
# degree is for mentioning the degree of polynomial. by default set to 2.
polynomial_converter.fit(X)

poly_trans = polynomial_converter.transform(X)
print(poly_trans)
print("\n")
print(poly_trans.shape)

print("\n")
print(poly_trans[0])

print("\n")
print(X.iloc[0])

# first three values of poly_trans are the original vals and the other 6 comprise of 3 of square and 3 of cube degrees.

# Generally fit_transform does both fit and transform simultaneously.
poly_trans_hybrid = polynomial_converter.fit_transform(X)
print(poly_trans_hybrid[0])
# Fit does no shit, until you do transform.

# Morbin' time :
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(poly_trans,y ,train_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

test_preds = model.predict(X_test)
print(test_preds)
print("\n")
# plt.plot(colour = 'red')

from sklearn.metrics import mean_squared_error,mean_absolute_error

mae = mean_absolute_error(y_test,test_preds)
mse = mean_absolute_error(y_test,test_preds)
rmse = np.sqrt(mse)

print(mae)
print("\n")
print(rmse)
print("\n")

print(rmse - mae)
print("\n")
# rmse >> mae
# rmse punishes the model at few data points by a lot....

print(model.coef_)






plt.show()