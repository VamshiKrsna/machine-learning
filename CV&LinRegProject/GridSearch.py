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

from sklearn.linear_model import ElasticNet

# Using ElasticNet for multi hyper parameter tuning

base_elastic_net_model = ElasticNet()

param_grid = { 'alpha' :  [0.1,1,5,10,50,100], 'l1_ratio' : [.1,.5,.7,.95,.99,1]}

from sklearn.model_selection import GridSearchCV

grid_model = GridSearchCV(param_grid=param_grid,estimator=base_elastic_net_model,
                          scoring='neg_mean_squared_error',cv = 5,
                          verbose=2)

print(grid_model.fit(X_train,y_train))

print(grid_model.best_estimator_)
# Returns best parameters for the data.

print(grid_model.cv_results_)

y_pres = grid_model.predict(X_test)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test,y_pres))


