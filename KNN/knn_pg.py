import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('gene_expression.csv')

print(df.head(5))

# sns.scatterplot(data=df,x= 'Gene One',y = 'Gene Two',hue = 'Cancer Present',
#                 alpha = 0.2,style= 'Cancer Present')

print(len(df))

# sns.pairplot(data = df,hue = 'Cancer Present')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Cancer Present',axis = 1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                    random_state=101)

scaler = StandardScaler()

sc_x_train = scaler.fit_transform(X_train)
sc_x_test = scaler.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=1)

knn_model.fit(sc_x_train,y_train)

y_preds = knn_model.predict(sc_x_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,y_preds))
print(len(y_test))
print(classification_report(y_test,y_preds))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_preds))

test_err_rate = []

for k in range(1,31):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(sc_x_train,y_train)

    ypreds_test = knn_model.predict(sc_x_test)

    test_err = 1 - accuracy_score(y_test,ypreds_test)
    test_err_rate.append(test_err) # appending to list, refers to index.

print(accuracy_score(y_test,ypreds_test))
print(test_err_rate)
print(max(test_err_rate))
print(min(test_err_rate))
# print("Ideal K value :")
# print(test_err_rate.index(0.06333333333333335)) # K value = index.
# # K = 9 gives min. test_error_rate.
# # so , k = 9 is ideal.
# print(test_err_rate.index(0.10444444444444445))
# This is not a best way to choose k .

# Plottin' time...

plt.plot(range(1,31),test_err_rate)
plt.ylabel("Error Rate")
plt.xlabel("K Vals")

# PIPELINING method for getting an ideal 'k'.

# PIPELINE --> GRIDSEARCH CV
knn = KNeighborsClassifier()

print(knn.get_params().keys())

ops = [('scaler',scaler),('knn',knn)]

from sklearn.pipeline import Pipeline
pipe = Pipeline(steps=ops)
from sklearn.model_selection import GridSearchCV

k_vals = list(range(1,31))

print(k_vals)

# param_grid is different for gridsearchcv
# Pay attention closely...
param_grid = {'knn__n_neighbors':k_vals}

full_cv_classifier = GridSearchCV(pipe,param_grid=param_grid,cv = 5,
                                  scoring='accuracy')

full_cv_classifier.fit(X_train,y_train)
# Pipeline contains scaler object. so pass in original data.

print(full_cv_classifier.best_estimator_.get_params())
# Look for knn__n_neighbors value.
# Looks like it has chosen 21 as best k value.

# Running preds and classification report.
full_y_preds = full_cv_classifier.predict(X_test)
print(classification_report(y_test,full_y_preds))
# Higher precision...(little, but better).

new_patient = [[3.8,6.4]]
print(full_cv_classifier.predict(new_patient))
print(full_cv_classifier.predict_proba(new_patient))

  knn_model2 = KNeighborsClassifier(n_neighbors=21)
knn_model2.fit(sc_x_train,y_train)
y_preds2 = knn_model2.predict(sc_x_test)
print(classification_report(y_test,y_preds2))
print(knn_model2.predict(new_patient))
print(knn_model2.predict_proba(new_patient)) # Really bad probabilites.

plt.show()