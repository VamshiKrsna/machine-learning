import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Rizen3\\Desktop\\vamshi\\VamshiProgrammingStuff\\PythonDS&ML\\PythonPDJP2021\\DATA\\Advertising.csv")

from sklearn.model_selection import train_test_split

X = df.drop('sales',axis = 1)
y = df['sales']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)

print(df.head(5))

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train,y_train)

test_predictions = model.predict(X_test)
print(test_predictions)
from sklearn.metrics import mean_absolute_error,mean_squared_error
print("\n")
print(df['sales'].mean())

# sns.histplot(data= df,x = 'sales')
print("\n")

print(mean_absolute_error(y_test,test_predictions))
print(mean_squared_error(y_test,test_predictions))
print("\n")

# Root Mean Square Error :
print(np.sqrt(mean_squared_error(y_test,test_predictions)))
print("\n")

# Residual Plots :

print(y_test - test_predictions) # These are the residuals.

test_residuals = y_test - test_predictions

sns.scatterplot(x = y_test,y =test_residuals)
# lets complete the residual plot with dotted red lines :
plt.axhline(y = 0,ls = '--',color = 'red')
# Voila , now we have our residual plot
# Note : a residual plot should look random and should not contain any parabolic structures.
# If it does , the data is not suitable for Linear regression.

sns.displot(test_residuals,kde = True,bins = 20)

plt.show()