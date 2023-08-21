import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

with open("Ames_Housing_Feature_Description.txt") as f:
    print(f.read())

df = pd.read_csv("Ames_outliers_removed.csv")
print(df.info())

print(df.head(5))
print(df.shape)
# Lets drop PID as it is useless for training
df = df.drop("PID",axis=1)
print(df.shape)
print(df.head(5))

print(df.isnull().sum()) # Sum will add all null value counts.
print("\n")
# we can see Lot Frontage has 490 null values
# sns.scatterplot(data = df,x = 'Gr Liv Area',y = 'SalePrice')
print(100*df.isnull().sum()/ len(df)) # Percentage of missing data by columns
print("\n")

def percent_missing(df):
    percent_nan = (100 * df.isnull().sum()/len(df))
    percent_nan = percent_nan[percent_nan > 0].sort_values()

    return percent_nan

percent_nan = percent_missing(df)
print(percent_nan)
print("\n")
sns.barplot(x = percent_nan.index, y = percent_nan)
plt.xticks(rotation = 90)

# Lets say we have 1 % of data is missing in a row, then it does not matter
# even if we drop those 1% data.



# Lets plot barplot in 0 to 1 percent range.
sns.barplot(x = percent_nan.index, y = percent_nan)
plt.xticks(rotation = 90)

print(percent_nan[percent_nan < 1])

# To drop one particular row containing null value in one category ,
# Follow :


# Lets drop electrical and garage cars missing data columns as they can effect
# others too.

df = df.dropna(axis = 0,subset=['Electrical','Garage Cars'])

print("\n")

print(df[df['Electrical'].isnull()]) # Empty
print(df[df['Garage Cars'].isnull()]) # Empty


# Lets compare bsmt full and half bath values if they're same :
print(df[df['Bsmt Full Bath'].isnull()])
print(df[df['Bsmt Half Bath'].isnull()])
# print(df[df['Bsmt Full Bath'].isnull()] == df[df['Bsmt Half Bath'].isnull()])
# Both are same.
bsmt_num_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF','Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']
df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)
bsmt_str_cols =  ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[bsmt_str_cols] = df[bsmt_str_cols].fillna('None')
percent_nan = percent_missing(df)
# sns.barplot(x=percent_nan.index,y=percent_nan)
# plt.xticks(rotation=90);
# plt.ylim(0,1)
percent_nan_data_now = percent_missing(df)
df["Mas Vnr Type"] = df["Mas Vnr Type"].fillna("None")
df["Mas Vnr Area"] = df["Mas Vnr Area"].fillna(0)
percent_nan = percent_missing(df)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);

gar_str_cols = ['Garage Type','Garage Yr Blt','Garage Finish','Garage Qual','Garage Cond']

df[gar_str_cols] = df[gar_str_cols].fillna(0)

percent_nan = percent_missing(df)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);

df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);


df = df.drop(['Fence','Alley','Misc Feature','Pool QC'],axis=1)

percent_nan = percent_missing(df)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);

print(df['Fireplace Qu'].value_counts())

df['Fireplace Qu'] = df['Fireplace Qu'].fillna("None")

# Lets fill fireplace qu by mean strategy :
df.transform(df['Fireplace Qu'])

plt.show()