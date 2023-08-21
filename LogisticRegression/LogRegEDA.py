# Data Analysis and Visualization only.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:\\Users\\Rizen3\\Desktop\\vamshi\\VamshiProgrammingStuff\\PythonDS&ML\\PythonPDJP2021\\DATA\\hearing_test.csv")

print(df.head(5))

print(df.describe())

print(df.shape)

print(df['test_result'].value_counts())# 3000 passed, 2000 didn't

sns.countplot(df['test_result'])  # Graphical Rep. of people that passed
plt.figure(dpi=150)
sns.boxplot(x = 'test_result',y = 'physical_score',data = df)
# ^ Relation b/w test_result and phy_score or age.

plt.figure(dpi= 150)
sns.scatterplot(x = 'age',y = 'physical_score',data = df,hue = 'test_result')
# ^ Relation of age vs phy_score and the ones passed and others who didn't

# plt.figure(dpi=150)
sns.pairplot(df,hue = 'test_result')
# ^ Pairplots

plt.figure(dpi=150)
sns.heatmap(df.corr(),annot = True)
# A heatmap showing the relation between all the variables from the data.

plt.figure(dpi=150)
sns.scatterplot(x = 'physical_score',y = 'test_result',data = df,hue='test_result')

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')

ax.scatter(df['age'],df['physical_score'],df['test_result'],c = df['test_result'])

plt.show()