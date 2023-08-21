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

from sklearn.metrics import mean_squared_error,mean_absolute_error

# Linear Regression Part :

from sklearn.linear_model import LinearRegression

lin_model = LinearRegression()
lin_model.fit(X_train,y_train)
lin_test_pred = lin_model.predict(X_test)

lin_mae = mean_absolute_error(y_test,lin_test_pred)
lin_mse = mean_squared_error(y_test,lin_test_pred)
lin_rmse = np.sqrt(lin_mse)

# Ridge Regression :

from sklearn.linear_model import RidgeCV

ridge_cv_model = RidgeCV(alphas=(0.1,1.0,10))
ridge_cv_model.fit(X_train,y_train)
ridge_test_pred = ridge_cv_model.predict(X_test)

ridge_mae = mean_absolute_error(y_test,ridge_test_pred)
ridge_mse = mean_squared_error(y_test,ridge_test_pred)
ridge_rmse = np.sqrt(ridge_mse)

# Lasso Regression :

from sklearn.linear_model import LassoCV

lasso_cv_model = LassoCV(eps=0.1,n_alphas=100,cv=5)
lasso_cv_model.fit(X_train,y_train)
lasso_test_preds = lasso_cv_model.predict(X_test)

lasso_mae = mean_absolute_error(y_test,lasso_test_preds)
lasso_mse = mean_squared_error(y_test,lasso_test_preds)
lasso_rmse = np.sqrt(lasso_mse)

from sklearn.linear_model import ElasticNetCV

elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                             eps = 0.001,n_alphas=100,max_iter = 1000)

elastic_model.fit(X_train,y_train)

# print(elastic_model.l1_ratio_) # Best one out of l1_ratio vals.
# 1.0 being the best indicates that it is l1 model 100%
print("\n")
print(elastic_model.alpha_)
print(lasso_cv_model.alpha_)

# since our model is l1 dominant, we can see concurrent alpha values for
# both elastic net and l1 or lasso model.

elastic_model_test_preds = elastic_model.predict(X_test)

elastic_model_mae = mean_absolute_error(y_test,elastic_model_test_preds)
elastic_model_mse = mean_squared_error(y_test,elastic_model_test_preds)
elastic_model_rmse = np.sqrt(elastic_model_mse)
print("\n")
print("lasso mae : ",lasso_mae)
print("elastic mae:",elastic_model_mae)
# lasso and elastic mae are so near.
# Therefore we can conclude that our lasso model and elastic net are so
# near to same.
