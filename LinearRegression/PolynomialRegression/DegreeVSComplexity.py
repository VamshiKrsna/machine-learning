import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Rizen3\\Desktop\\vamshi\\VamshiProgrammingStuff\\PythonDS&ML\\PythonPDJP2021\\DATA\\Advertising.csv")

X = df.drop('sales',axis = 1)
y = df['sales']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

# Choosing Degree of Polynomial

train_rmse_err = []
test_rmse_err = []


for d in range(1,10):

    polynomial_convert = PolynomialFeatures(degree=d,include_bias=False)
    poly_features = polynomial_convert.fit_transform(X)

    X_train , X_test, y_train, y_test = train_test_split(poly_features,y,test_size= 0.33)

    model = LinearRegression()
    model.fit(X_train,y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train,train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test,test_pred))

    train_rmse_err.append(train_rmse)
    test_rmse_err.append(test_rmse)

    print(f"train rmse of degree ",d," is : ",train_rmse)
    print(f"test rmse of degree  ",d," is : ",test_rmse)

    print("\n")

print(train_rmse_err)
print(test_rmse_err)

# plt.plot(range(1,6),train_rmse_err[:5],label = 'Train rmse')
# plt.plot(range(1,6),test_rmse_err[:5],label = 'Test rmse')
#
# plt.xlabel("Complexity")
# plt.ylabel("Degree(error)")

