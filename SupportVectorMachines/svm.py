import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('mouse_viral_study.csv')
print(df.head(5))

# sns.scatterplot(data = df,x='Med_1_mL',y = 'Med_2_mL',
#                 hue = 'Virus Present')

# Simple Classifier Line.

x = np.linspace(0,10,100)
m = -1
b = 11
y = (m * x) + b
plt.plot(x,y,'black')

from sklearn.svm import SVC

print(help(SVC))

y = df['Virus Present']
X = df.drop('Virus Present',axis= 1)

model = SVC(kernel="linear",C=1000)
# C is the punishment parameter.
# Larger the C , More is the penalty, Stricter the evaluation.
model.fit(X,y)

from svm_margin_plot import plot_svm_boundary
# plot_svm_boundary(model,X,y)

# The above module is taken directly from documentation.

# model2 = SVC(kernel='rbf',C = 0.05)
# model2.fit(X,y)
# plot_svm_boundary(model2,X,y)

model3 = SVC(kernel='rbf',C = 1)
# rbf = Radial Basis Function ...
# model3.fit(X,y)
# plot_svm_boundary(model3,X,y)

# Sigmoid Kernel

# modelsig = SVC(kernel='sigmoid')
# modelsig.fit(X,y)
# plot_svm_boundary(modelsig,X,y)

# Polynomial Kernel :

modelpoly = SVC(kernel='poly',C=1,degree=2)
modelpoly.fit(X,y)
# plot_svm_boundary(modelpoly,X,y)

# Tuning :

from sklearn.model_selection import GridSearchCV

final = SVC()
param_grid = {"C":[0.01,0.1,1],"kernel":['linear','rbf']}

grid = GridSearchCV(final,param_grid)
grid.fit(X,y)
print(grid.best_params_)

# plt.show()