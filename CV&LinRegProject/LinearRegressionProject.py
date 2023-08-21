import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Rizen3\\Desktop\\vamshi\\VamshiProgrammingStuff\\PythonDS&ML\\PythonPDJP2021\\DATA\\AMES_Final_DF.csv")

print(df.info())
print(df.head(5))

# As we are predicting SalePrice Column :

X = df.drop('SalePrice',axis=1)
y = df['SalePrice']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import ElasticNet

base_model = ElasticNet()

param_grid = {'alpha' : [0.1,0.01,1,10,100],
              'l1_ratio' : [0.1,0.25,0.50,0.75,0.99,1.0]}

from sklearn.model_selection import GridSearchCV

gs_model = GridSearchCV(estimator=base_model,param_grid=param_grid,
                        cv = 5, verbose=1)

gs_model.fit(X_train,y_train)

print(gs_model.best_params_)

y_preds = gs_model.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error

print("MAE = ",mean_absolute_error(y_test,y_preds))
print("RMSE = ",math.sqrt(mean_squared_error(y_test,y_preds)))

# Lets see actual mean SalePrice of houses in our df :

MeanPriceOfHouses = mean_absolute_error(y_test,y_preds)
ActualMeanPriceOfHouses = np.mean(df['SalePrice'])

DiffOfModel = ActualMeanPriceOfHouses - MeanPriceOfHouses

print("Actual Price of Houses(Mean) = ",ActualMeanPriceOfHouses)
print("Mean Price of Houses(Model) = ",MeanPriceOfHouses)
print("Difference in Actual and model : ",DiffOfModel)