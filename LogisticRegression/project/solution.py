import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart.csv")
print(df.head(5))
print(df.shape)

print(df.describe())
print(df.info())
# sns.countplot(data=df,x = 'target')

# sns.pairplot(df[['age','chol','thalach','target']]
             # ,hue='target')
# fig= plt.figure(dpi = 150,figsize=(200,100))
# sns.heatmap(data = df.corr(),annot = True,cmap = 'viridis')

X = df.drop('target',axis = 1)
y = df['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                    test_size=0.10, random_state=101)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV


log_model = LogisticRegressionCV()
# penalty = ['l1','l2','elasticnet']
# l1_ratio = np.linspace(0,1,20)
# C = np.logspace(0,10,20)
# param_grid = {
#     'penalty':penalty,
#     'l1_ratio':l1_ratio,
#     'C':C
#     }
# grid_model = GridSearchCV(log_model,param_grid=param_grid)
log_model.fit(scaled_X_train,y_train)

print(log_model)

print(log_model.get_params())
print(log_model.Cs_)
print(log_model.C_) # Best C value out of all.
print(log_model.coef_)

coefs = pd.Series(data=log_model.coef_[0],index=X.columns)

# print(coefs)
coefs = coefs.sort_values() # Ascending order.
print(coefs)

fig = plt.figure(figsize=(10,6),dpi = 100)
sns.barplot(x = coefs.index,y = coefs.values)

# Metric Evaluation :
from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix,plot_roc_curve,confusion_matrix,plot_precision_recall_curve

y_preds = log_model.predict(scaled_X_test)
print(confusion_matrix(y_test,y_preds))
# plot_confusion_matrix(log_model,scaled_X_test,y_test)
print(classification_report(y_test,y_preds))
# plot_precision_recall_curve(log_model,scaled_X_test,y_test)
# plot_roc_curve(log_model,scaled_X_test,y_test)

print(accuracy_score(y_preds,y_test) * 100)
# Prediction of some other patient

patient1 = [[54.,1.,0.,122.,286.,0.,0.,116.,1.,3.2,1.,2.,2.]]
patient2 = [[20,1,2,110,230,1,1,140,1,2.2,2,0,2]]
patient3 = [[40,1,2,150,270,1,1,180,1,2.2,2,0,2]]


print(log_model.predict(patient1))
print(log_model.predict(patient2))
print(log_model.predict(patient3))

print(log_model.predict_proba(patient1))
confid1 = log_model.predict_proba(patient1)[0][0]
print("confidence :")
print(confid1)


print(log_model.predict_proba(patient2))
confid_2 = log_model.predict_proba(patient2)[0][1]
print("confidence :")
print(confid_2)

print(log_model.predict_proba(patient3))
confid_3 = log_model.predict_proba(patient3)[0][1]
print("confidence :")
print(confid_3)


# 99.999 percent sure that the prediction is correct.
plt.show()