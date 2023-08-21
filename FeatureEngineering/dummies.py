import numpy as np
import pandas as pd

mystr = pd.Series(list('abcdabc'))
print(pd.get_dummies(mystr))
print("\n")
s1 = ['a','b','c',np.nan]
print(pd.get_dummies(s1))