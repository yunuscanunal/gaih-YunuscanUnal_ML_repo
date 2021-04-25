#!/usr/bin/env python
# coding: utf-8

# ##### sklearn.datasets sınıfında "make_classification" fonksiyonunu kullanarak veri kümesi oluşturun. 
# - Tek etiket (y) 9 özellik (X) ile 10000 örnek oluşturun. Ayrıca şu parametreleri kullanın:
# - n_informative = 4
# - class_sep = 2
# - random_state = 18

# Gerekli kitaplıkları içe aktarın.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Sklearn'de make_classification fonksiyonunu kullanarak veri kümesi oluşturun.
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10000, n_features=9, n_informative=4, class_sep=2, random_state=18)
print("Shape of X: ", X.shape,"\nShape of y: ", y.shape)



df = pd.DataFrame(X, columns= ("x1","x2","x3","x4","x5","x6","x7","x8","x9"))
y = pd.DataFrame(y)


# Yinelenen satırları ve eksik verileri kontrol edin.
print("Number of Duplicated Data: ", df.duplicated().sum())
print("\nSum of NA Data:\n", df.isna().sum())



# Her özellik için verileri görselleştirin (pairplot, distplot etc.)
sns.pairplot(df)



# Korelasyon matrisi çizin.
corr = df.corr()

plt.figure(figsize=(15,15))
sns.heatmap(corr, annot=True, linewidths=.5, cmap="BuPu")



#Drop x4 and x5 features because of high correlation
df = df.drop(["x4","x5"] ,axis=1).reset_index(drop= True)
df


# Aykırı değerleri işleyin (IsolationForest, Z-score, IQR kullanabilirsiniz)
from scipy import stats
z = np.abs(stats.zscore(df))
print("length of anomaly list: ", len(np.where(z>3)[0]))
print("Index of Anomaly data:", np.where(z>3)[0])

outliers = list(set(np.where(z>3)[0]))
new_df = df.drop(outliers, axis=0).reset_index(drop= False)

new_y = y.drop(index=outliers, axis=0)



new_df = new_df.drop(["index"] ,axis=1).reset_index(drop= True)
new_df


# Veri kümesini eğitim ve test verisetlerine ayırın.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_df, new_y, test_size=0.3, random_state=42)


# Karar Ağacını içe aktarın, farklı hiperparametreleri deneyerek algoritmayı ayarlayın. (hyperpara)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth= 2, random_state= 22)
clf.fit(X_train, y_train)
print("Accuracy of train: ", clf.score(X_train, y_train))
print("Accuracy of test: ", clf.score(X_test, y_test))

# Öznitelik önemlerini (feature importances) görselleştirin.
plt.figure(figsize=(20,20))
importances = clf.feature_importances_
sns.barplot(x= importances, y= X_test.columns)
plt.show()


# Hata matrisini oluşturun ve accuracy, recall, precision ve f1-score değerlerini hesaplayın.
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score
pred = clf.predict(X_test)
print(classification_report(y_test,pred))
print("*"*100,"\n")

# Metrics
print("Precision = {}".format(precision_score(y_test, pred, average='macro')))
print("Recall = {}".format(recall_score(y_test, pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, pred)))
print("F1 Score = {}\n".format(f1_score(y_test, pred,average='macro')))
print("*"*100,"\n")
print("*"*100,"\n")

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(12, 8))
ax =sns.heatmap(cm,fmt="f", square=True, annot=True, cbar=False)
#ax.xaxis.set_ticklabels(categories, fontsize = 12)
#ax.yaxis.set_ticklabels(categories, fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()


get_ipython().system('pip install xgboost')
import xgboost as xgb

# XGBoostClassifier'ı içe aktarın, farklı hiperparametreleri deneyerek algoritmayı ayarlayın.

from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_dict = {
    'max_depth':range(2,5,2),
    'min_child_weight':range(1,5,2),
    'learning_rate': [0.001,0.01],
    'n_estimators': [10,190,210,500],
    'num_class': [3]
    
}

xgc = XGBClassifier(booster='gbtree', learning_rate =0.01, n_estimators=200, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softprob', nthread=4, scale_pos_weight=1, seed=27)

clf = GridSearchCV(xgc, param_dict, cv=4, n_jobs = -1).fit(X_train,y_train)

print("Tuned: {}".format(clf.best_params_)) 
print("Mean of the cv scores is {:.6f}".format(clf.best_score_))
print("Train Score {:.6f}".format(clf.score(X_train,y_train)))
print("Test Score {:.6f}".format(clf.score(X_test,y_test)))
print("Seconds used for refitting the best model on the train dataset: {:.6f}".format(clf.refit_time_))


# Öznitelik önemlerini (feature importances) görselleştirin.
xgc = XGBClassifier(booster='gbtree', learning_rate =0.01, n_estimators=500, max_depth=4,
 min_child_weight=1, num_class=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softprob', nthread=4, scale_pos_weight=1, seed=27)
xgc.fit(X_train, y_train)


plt.figure(figsize=(20,20))
importances2 = xgc.feature_importances_
sns.barplot(x= importances2, y= new_df.columns)
plt.show()


# Hata matrisini oluşturun ve accuracy, recall, precision ve f1-score değerlerini hesaplayın.
pred = xgc.predict(X_test)
print(classification_report(y_test,pred))
print("*"*100,"\n")

# Metrics
print("Precision = {}".format(precision_score(y_test, pred, average='macro')))
print("Recall = {}".format(recall_score(y_test, pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, pred)))
print("F1 Score = {}\n".format(f1_score(y_test, pred,average='macro')))
print("*"*100,"\n")
print("*"*100,"\n")

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(12, 8))
ax =sns.heatmap(cm, fmt="f", square=True, annot=True, cbar=False)
#ax.xaxis.set_ticklabels(categories, fontsize = 12)
#ax.yaxis.set_ticklabels(categories, fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()


# Sonucunuzu değerlendirin ve veri setimiz için en iyi performans gösteren algoritmayı seçin.


# - Modelimiz %98 başarı gösterdi ve oldukça başarılı bir model skoru diyebiliriz.
# - Eğittiğimiz modellerden karar ağacı modelimiz (Decision Tree) %94 başarı sağladı.
# - Daha sonra xgboost modelimiz %98 başarı sağladı.
# - xgboost modelimiz için hyperparams denemesi yaptık.
# - Parametre denememizde de çıkan sonuçlara göre modelimizin en iyi parametleri:
#  + learning_rate: 0.01
#  + max_depth: 4
#  + min_child_weight: 1
#  + n_estimators: 500
# - Hyperparams denememizden sonra:
#  + 0 tahmin ettiğimiz fakat gerçek değeri 1 olan verilerimizin sayısı 100'den 28'e düştü.
#  + 1 tahmin ettiğimiz fakat gerçek değeri 0 olan verilerimizin sayısı 75'ten 31'e düştü.
#  + Doğru tahmin ettiğimiz verilerimizin sayısı 2788'den 2904'e çıktı.
#  + Yanlış tahmin ettiğimiz verilerimizin sayısı 175'ten 59'a düştü
