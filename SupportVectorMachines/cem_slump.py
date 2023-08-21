import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cement_slump.csv")

print(df.head(5))
print(df.columns)

# plt.figure(figsize=(6,6),dpi=200)
# print(sns.heatmap(df.corr(),annot = True))

X = df.drop('Compressive Strength (28-day)(Mpa)',axis=1)
y = df['Compressive Strength (28-day)(Mpa)']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from  sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

from sklearn.svm import SVC,SVR,LinearSVR

# help(SVR)

base_model = SVR()

base_model.fit(scaled_x_train,y_train)

base_preds = base_model.predict(scaled_x_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error

print("MAE :",mean_absolute_error(y_test,base_preds))
print("RMSE:",np.sqrt(mean_squared_error(y_test,base_preds)))

from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.001,0.01,0.1,1,0.5],
              'kernel':['linear','rbf','poly'],
              'gamma':['scale','auto'],
              'degree':[2,3,4],
              'epsilon':[0,0.01,0.1,0.5,1,2]}

svr = SVR()

grid = GridSearchCV(svr,param_grid)
grid.fit(scaled_x_train,y_train)

# print(grid.best_params_)
#{'C': 1, 'degree': 2, 'epsilon': 2, 'gamma': 'scale', 'kernel': 'linear'}
grid_preds = grid.predict(scaled_x_test)

print("MAE :",mean_absolute_error(y_test,grid_preds))
print("RMSE:",np.sqrt(mean_squared_error(y_test,grid_preds)))


best_svr = SVR(C= 1,degree=2,epsilon=2,gamma='scale',kernel='linear')
best_svr.fit(scaled_x_train,y_train)
preds = best_svr.predict(scaled_x_test)

print(mean_absolute_error(y_test,preds))

# best_svr.predict()


plt.show()