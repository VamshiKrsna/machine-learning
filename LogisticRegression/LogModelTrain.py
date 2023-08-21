import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:\\Users\\Rizen3\\Desktop\\vamshi\\VamshiProgrammingStuff\\PythonDS&ML\\PythonPDJP2021\\DATA\\hearing_test.csv")

print(df.head(5))

X = df.drop('test_result',axis = 1)
y = df['test_result']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()

log_model.fit(scaled_X_train,y_train)

print(log_model.coef_)

y_prob_preds = log_model.predict_proba(scaled_X_test)

print(y_prob_preds)

print(log_model.coef_)

from sklearn.metrics import accuracy_score,confusion_matrix,\
    classification_report

print(y_test)
y_preds = log_model.predict(scaled_X_test)
print(y_preds)

print(accuracy_score(y_test,y_preds)*100)

conf_mat = confusion_matrix(y_test,y_preds)
print(conf_mat)
print(conf_mat)

tn,fp,fn,tp = conf_mat.ravel()

print((tn,fp,fn,tp))

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(log_model,scaled_X_test,y_test)

print(classification_report(y_test,y_preds))

from sklearn.metrics import precision_score,recall_score

print(precision_score(y_test,y_preds)*100)
print(recall_score(y_test,y_preds)*100)

from sklearn.metrics import plot_precision_recall_curve,plot_roc_curve

fig, ax = plt.subplots(figsize=(12,8),dpi = 100)
plot_roc_curve(log_model,scaled_X_test,y_test,ax=ax)

plot_precision_recall_curve(log_model,scaled_X_test,y_test)

plt.show()