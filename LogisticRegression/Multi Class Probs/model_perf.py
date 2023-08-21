import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
print(df.head(5))

X = df.drop('species',axis=1)
y = df['species']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y,
                    test_size=0.25, random_state=101)

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

log_model = LogisticRegression(solver='saga',multi_class='ovr',max_iter=1000)
# ovr = one vs rest
# sag = stochastic avg gradient descent
# saga = "          "    "

penalty = ['l1','l2','elasticnet']
l1_ratio = np.linspace(0,1,20)
C = np.logspace(0,10,20)

param_grid = {'penalty': penalty,
              'l1_ratio':l1_ratio,
              'C':C}

grid_model = GridSearchCV(log_model,param_grid=param_grid)

grid_model.fit(scaled_X_train,y_train)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
print(grid_model.best_params_)
# print("Banana")
y_pred = grid_model.predict(scaled_X_test)
print(accuracy_score(y_test,y_pred)*100)

print(confusion_matrix(y_test,y_pred))
plot_confusion_matrix(grid_model,scaled_X_test,y_test)

print(classification_report(y_test,y_pred))

plt.show()