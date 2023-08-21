import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Ames_NO_Missing_Data.csv")

print(df.shape)
# print(df.isnull().sum())

df['MS SubClass'] = df['MS SubClass'].apply(str)
print(df['MS SubClass'])

direction = pd.Series(['Up','Up','Down'])
print(direction)
print(pd.get_dummies(direction))
print(pd.get_dummies(direction,drop_first=True)) # Generally how get_dummies works...

df.select_dtypes(include= 'object') # selects only object data...
print(df.info()) # Info of dataframe.

object_data = df.select_dtypes(include= 'object')
numerical_data = df.select_dtypes(exclude='object')

df_object_dummies = pd.get_dummies(object_data,drop_first=True)
print(df_object_dummies)
# We can see 238 columns whereas original df has only 76.
# Excess columns are various one hot encoded dummies.

# No need to operate on numerical data as it has no categories.
# Technically , years are directly stated and dont have any encoding.
# so , ggs.

final_df = pd.concat([numerical_data,df_object_dummies],axis=1)

print(final_df)
