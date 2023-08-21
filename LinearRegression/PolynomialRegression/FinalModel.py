import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
# Final Model :

df = pd.read_csv("C:\\Users\\Rizen3\\Desktop\\vamshi\\VamshiProgrammingStuff\\PythonDS&ML\\PythonPDJP2021\\DATA\\Advertising.csv")

X = df.drop('sales',axis = 1)
y = df['sales']


final_poly_converter = PolynomialFeatures()
final_model = LinearRegression()

full_converted_X = final_poly_converter.fit_transform(X)
final_model.fit(full_converted_X,y)

from joblib import dump,load
dump(final_model,'final_poly_model.joblib')
dump(final_poly_converter,'final_converter.joblib')

loaded_converter = load('final_converter.joblib')
loaded_model = load('final_poly_model.joblib')

campaign = [[149,22,12]]

print(loaded_converter.fit_transform(campaign))
transformed_data = loaded_converter.fit_transform(campaign)
print("\n")
print(loaded_model.predict(transformed_data))
