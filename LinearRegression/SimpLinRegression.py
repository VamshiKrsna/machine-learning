import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Advertising.csv")

print(df.head(5))

df['total_expend'] = df['TV'] + df['radio'] + df['newspaper']

print(df.head())

# Here we have to find the relation bw advertising and sales ;

# sns.scatterplot(data = df, x = 'total_expend',y = 'sales')

# The more you advertise , the more you get sales (Linear).
# Regression plot :
sns.regplot(data = df,x = 'total_expend',y = 'sales')
# regplot gives us the best plot with best regression line.

X = df['total_expend']
y = df['sales']

# help(np.polyfit) # Least Squares Polynomial fit function
ols_df = np.polyfit(X,y,deg=1) # Deg 1 is to specify Bn for n = 1,0
print(ols_df)

# y = mx + b
# y = B1(x) + B0

B1 = ols_df [0]
B0 = ols_df [1]

potential_expend = np.linspace(0,500,100)
potential_spend = B1 * potential_expend + B0

# plt.figure(figsize=(10,4),dpi=150)
plt.plot(potential_expend,potential_spend,color = 'red') # This red line is the regression line.

spend = 200
predicted_sales = B1 * spend + B0
print(predicted_sales) # 13.98 This is predicted
print()

# y = B1x + B0
# y = B3x^3 + B2x

plt.show()
