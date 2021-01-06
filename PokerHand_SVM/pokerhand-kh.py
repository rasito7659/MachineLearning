#import library
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd

#memanggil data
df = pd.read_csv('poker-hand-training-true.data.txt')

#menyimpan dataset pada variabel (X = attribute, y = class)
X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

#membagi dataset dengan rasio train:test = 4 : 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#membuat klasifikasi svm
clf = svm.SVC(kernel='rbf', gamma='auto')
#training data
clf.fit(X_train, y_train)

#prediksi data dari dataset
y_pred = clf.predict(X_test)

#import library untuk menghitung akurasi
from sklearn import metrics
#penghitungan akurasi
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:")
print(accuracy)
print("Precision, Recall, F-Measure:")
result = metrics.precision_recall_fscore_support(y_test, y_pred, average='micro')
print(result)

#testing prediksi dari data yang tidak ada pada dataset
testing = np.array([1,3,4,5,3,4,1,12,4,6])
print("New Atribute Value = 1,3,4,5,3,4,1,12,4,6")
testing = testing.reshape(1, -1)
prediction = clf.predict(testing)
print("Prediction Value:")
print(prediction)