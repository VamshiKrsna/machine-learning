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

# print(X_train[0])
# print( poly_feature[0])

print("\n")

from sklearn.linear_model import LassoCV

lasso_cv_model = LassoCV(eps=0.1,n_alphas=100,cv=5)
print(lasso_cv_model)
lasso_cv_model.fit(X_train,y_train)
print(lasso_cv_model.alpha_)  # Best alpha value.
# eps value will understand alpha min and alpha max and our n_alphas
# create n no. of alpha values.
#eps = alpha_min / alpha_max.

test_preds = lasso_cv_model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error
MAE = mean_absolute_error(y_test,test_preds)
MSE = mean_squared_error(y_test,test_preds)
RMSE = np.sqrt(MSE)

print("\n")

from sklearn.linear_model import LinearRegression

lintest = LinearRegression()
lintest.fit(X_train,y_train)

lintest_preds = lintest.predict(X_test)

from sklearn.linear_model import Ridge

ridtest = Ridge()
ridtest.fit(X_train,y_train)
ridtest_preds = ridtest.predict(X_test)

rid_mae = mean_absolute_error(y_test,ridtest_preds)
rid_rmse = np.sqrt(mean_squared_error(y_test,ridtest_preds))

ori_mae = mean_absolute_error(y_test,lintest_preds)
ori_rmse = np.sqrt(mean_squared_error(y_test,lintest_preds))

print("Original Test Preds MAE :",ori_mae)
print("Lasso MAE :",MAE)
print("Ridge MAE :",rid_mae)
print("Original Test Preds RMSE : ",ori_rmse)
print("Lasso RMSE :",RMSE)
print("Ridge RMSE :",rid_rmse)

print("\n")
print(lasso_cv_model.coef_)

i = 0

for feature in lasso_cv_model.coef_:
    if feature == 0. :
        i += 1
print(i ," Nil Features ...")
# We can see that Lasso is not performing well.

# param's can be a reason for this too.
