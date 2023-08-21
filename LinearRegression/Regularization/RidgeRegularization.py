import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
# Final Model :

df = pd.read_csv("C:\\Users\\Rizen3\\Desktop\\vamshi\\VamshiProgrammingStuff\\PythonDS&ML\\PythonPDJP2021\\DATA\\Advertising.csv")

X = df.drop('sales',axis = 1)
y = df['sales']  # Test Data . Never touch until end.

poly_converter = PolynomialFeatures(degree=3,include_bias=False)

poly_feature = poly_converter.fit_transform(X)
print(poly_feature.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(poly_feature,y,test_size=0.30,random_state=101)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

print(X_train[0])
print( poly_feature[0])

print("\n")
# End of Data Setup ....

# Ridge Regression
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=10)
ridge_model.fit(X_train,y_train)
test_pred = ridge_model.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error

MAE = mean_absolute_error(y_test,test_pred)
MSE = mean_squared_error(y_test,test_pred)
RMSE = np.sqrt(MSE)

print(  MAE ) # Original MAE
print(MSE )
print(RMSE)

from sklearn.linear_model import RidgeCV

ridge_cv_model = RidgeCV(alphas=(0.1,1.0,10)) # Ridge CV takes 3 alphas.
ridge_cv_model.fit(X_train,y_train)

# Lets see best perfroming alpha :
print(ridge_cv_model.alpha_) # We can see that 0.1 is the best alpha value.
# Out of 3 Alphas.

from sklearn.metrics import  SCORERS
print(SCORERS.keys()) # Components of Scorers

print("\n")

ridge_test_predictions = ridge_cv_model.predict(X_test)

ridge_mse = mean_squared_error(y_test,ridge_test_predictions)
ridge_rmse = np.sqrt(mean_absolute_error(y_test,ridge_test_predictions))

print(MSE)
print(ridge_mse)
print(MSE - ridge_mse)
print("\n")
print(RMSE)
print(ridge_rmse)
print(RMSE - ridge_rmse)
print("\n")

# We can conclude that the mse, rmse of the ridge test predictions is
# much lesser than original test predictions.

