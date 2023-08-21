import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_ages(mu = 50,sigma = 13,num_samples = 100,seed = 42):
    np.random.seed(seed)

    sample_ages = np.random.normal(loc=mu,scale=sigma,size=num_samples)
    sample_ages = np.round(sample_ages)

    return sample_ages

sample = create_ages()
print(sample)

# sns.displot(sample,bins = 20)
# sns.boxplot(sample)

ser = pd.Series(sample)
print(ser.describe())
print(len(ser))
print("\\n")
# IQR : Inter Quartile Range Method :
# Refer Notes for Info.

IQR = 55.25 - 42.0
# IQR = 3rd Quartile - 1st Quartile.
# IQR = 75% - 25%
lower_bound = (42 - (IQR * 1.5))
upper_bound = (55.25 + (IQR * 1.5))

print(IQR)
print(lower_bound)
print(upper_bound)

ser = ser[ser > lower_bound]
ser = ser[ser < upper_bound]

print(len(ser))
print("\\n")

# More easy and Robust way.

q3,q1 =np.percentile(sample,[75,25]) # another way to print IQR Q1 and Q3
iqr = q3-q1
lb = q1 - (1.5 * iqr)
ub = q3 + (1.5 * iqr)

print(iqr)
print(lb)
print(ub)

df = pd.read_csv("C:\\Users\\Rizen3\\Desktop\\vamshi\\VamshiProgrammingStuff\\PythonDS&ML\\PythonPDJP2021\\DATA\\Ames_Housing_Data.csv")
# print(df.head(5))
print(df.corr()['SalePrice'].sort_values())

# sns.scatterplot(x=df['Overall Qual'],y=df['SalePrice'])

# Lets filter some outliers.
# Domain Knowledge flex.
# as our dataset is related to real estate,
# we can see 3 datapoints at bottom with low price and high overall quality.
# which is unusual.
# lets compare Gr Liv Area and SalePrice of that houses based on observation.
sns.scatterplot(x = 'Gr Liv Area',y = 'SalePrice',data = df)
# We can see 3 homes with high living area and high SalePrice
# The top ones are not outliers. They indeed contribute to lin regression.
# Lets find them :
print(df[(df['Overall Qual'] > 8) & (df['SalePrice']<200000)])

# analyse and print them 3 houses :
print(df[(df['Gr Liv Area'] > 4000 )& (df['SalePrice'] < 400000)])
# The above are the outliers

print("\n")

drop_ind = df[(df['Gr Liv Area']>4000) & (df['SalePrice']<400000)]
df = df.drop(drop_ind,axis=0)
print(df)

df.to_csv('Ames_No_Outliers_final.csv')

sns.scatterplot(x = 'Gr Liv Area',y = 'SalePrice',data = df)

plt.show()