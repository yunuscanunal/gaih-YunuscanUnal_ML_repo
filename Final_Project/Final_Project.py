#!/usr/bin/env python
# coding: utf-8

# # Project
# 
# In this project, our aim is to building a model for predicting wine qualities. Our label will be `quality` column. Do not forget, this is a Classification problem!
# 
# ## Steps
# - Read the `winequality.csv` file and describe it.
# - Make at least 4 different analysis on Exploratory Data Analysis section.
# - Pre-process the dataset to get ready for ML application. (Check missing data and handle them, can we need to do scaling or feature extraction etc.)
# - Define appropriate evaluation metric for our case (classification).
# - Train and evaluate Decision Trees and at least 2 different appropriate algorithm which you can choose from scikit-learn library.
# - Is there any overfitting and underfitting? Interpret your results and try to overcome if there is any problem in a new section.
# - Create confusion metrics for each algorithm and display Accuracy, Recall, Precision and F1-Score values.
# - Analyse and compare results of 3 algorithms.
# - Select best performing model based on evaluation metric you chose on test dataset.
# 
# 
# Good luck :)

# <h2>Yunuscan Ünal</h2>

# # Data

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Read csv
df = pd.read_csv("winequality.csv")
df



# Describe our data for each feature and use .info() for get information about our dataset
df.info()
# Analyse missing values
print("*"*100)
print("Total NA Data: \n", df.isna().sum())


# # Exploratory Data Analysis

# Our label Distribution (countplot)
sns.countplot(df.quality)


# Example EDA (distplot)
sns.distplot(df["citric acid"])


sns.pairplot(df)


# # Preprocessing
# 
# - Are there any duplicated values?
# - Do we need to do feature scaling?
# - Do we need to generate new features?
# - Split Train and Test dataset. (0.7/0.3)

print("Total Duplicated Data: ", df.duplicated().sum())


df = df.drop_duplicates()
df



corr = df.corr()

plt.figure(figsize=(15,15))
sns.heatmap(corr, annot=True, linewidths=.5, cmap="BuPu")

df = df.drop(["citric acid","density","pH", "free sulfur dioxide"] ,axis=1).reset_index(drop= True)
df


from scipy import stats
z = np.abs(stats.zscore(df))
print("length of anomaly list: ", len(np.where(z>3)[0]))
print("Index of Anomaly data:", np.where(z>3)[0])

outliers = list(set(np.where(z>3)[0]))
new_df = df.drop(outliers, axis=0).reset_index(drop= True)


y = pd.DataFrame(new_df["quality"])
new_df = new_df.drop(columns=["quality"], axis=0)



from sklearn.preprocessing import StandardScaler, MinMaxScaler
X_scaled = StandardScaler().fit_transform(new_df)
X_scaled


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=22)


# # ML Application
# 
# - Define models.
# - Fit models.
# - Evaluate models for both train and test dataset.
# - Generate Confusion Matrix and scores of Accuracy, Recall, Precision and F1-Score.
# - Analyse occurrence of overfitting and underfitting. If there is any of them, try to overcome it within a different section.

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth= 3, random_state= 22)
clf.fit(X_train, y_train)
print("Accuracy of train: ", clf.score(X_train, y_train))
print("Accuracy of test: ", clf.score(X_test, y_test))

#plt.figure(figsize=(20,20))
importances = clf.feature_importances_
sns.barplot(x= importances, y= X_train.columns)
plt.show()


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
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth= 3,random_state=22)
rfc.fit(X_train, y_train)

print("Accuracy of train: ", rfc.score(X_train, y_train))
print("Accuracy of test: ", rfc.score(X_test, y_test))


importances = rfc.feature_importances_
sns.barplot(x= importances, y= X_train.columns)
plt.show()

pred = rfc.predict(X_test)
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

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(12, 8))
ax =sns.heatmap(cm, fmt="f", square=True, annot=True, cbar=False)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()


import xgboost as xgb
from xgboost.sklearn import XGBClassifier

xgc = XGBClassifier(max_depth= 3,random_state=22)
xgc.fit(X_train, y_train)

print("Accuracy of train: ", xgc.score(X_train, y_train))
print("Accuracy of test: ", xgc.score(X_test, y_test))


importances = xgc.feature_importances_
sns.barplot(x= importances, y= X_train.columns)
plt.show()


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

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(12, 8))
ax =sns.heatmap(cm, fmt="f", square=True, annot=True, cbar=False)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()


clf.set_params()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_dict = {
    'max_depth':range(1,10),
    'min_samples_leaf':range(1,20),
    'min_samples_split': range(1,20),
    'criterion': ["gini", "entropy"]
    
}

clf = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                             max_depth=3, max_features="log2", max_leaf_nodes=None,
                             min_impurity_decrease=0.0, min_impurity_split=None,
                             min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, presort='deprecated',
                             random_state=22, splitter='best')

clf2 = GridSearchCV(clf, param_dict, cv=4, n_jobs = -1).fit(X_train,y_train)

print("Tuned: {}".format(clf2.best_params_)) 
print("Mean of the cv scores is {:.6f}".format(clf2.best_score_))
print("Train Score {:.6f}".format(clf2.score(X_train,y_train)))
print("Test Score {:.6f}".format(clf2.score(X_test,y_test)))
print("Seconds used for refitting the best model on the train dataset: {:.6f}".format(clf2.refit_time_))


# # Evaluation
# 
# - Select the best performing model and write your comments about why choose this model.
#  + Decision tree modelimizin sonuçları: acc of test = 0.5968
#  + XGBoost modelimizin sonuçları: acc of test = 0.5756
#  + Random Forest modelimizin sonuçları: acc of test = 0.5809
#  + Sınıflandırma projesi olduğu için model eğitirken karar ağacı (Decision tree) modeli ile veri setini denedim. Daha sonrasında xgboost modelinden daha iyi performans alacağımı düşündüğüm için xgboost modelini eğittim. Ek olarak random forest modelini de denedim. Fakat en iyi test sonucumuz decision tree algoritmasıyla çıktı.
#  + Decision tree modelimize hyperparams yapmaya çalıştım. Train skorumuz artarken test skorumuz düştü. Bu da modelimizin overfitting sürecine girdiği anlamına gelebilir.
#  + En iyi sonuç decision tree modelinde max_depth: 3 iken geldi. 
#  
# - Analyse results and make comment about how you can improve model.
#  + %60lık bir başarı modelimizin yeterince başarısız olduğunu göstermektedir.
#  + Daha iyi hyperparams yapılabilir miydi araştırmak ve uygulamak gerekir.
#  + Daha iyi bir model eğitebilmek için veri setimizi büyütmeyi ve elimizdeki feature selection'ı daha iyi yapmayı denemeliyim.
#  + Farklı sınıflandırma modelleri deneyerek modellerin başarısını kıyaslamak ve en iyisini bulmak toplam başarımı arttırabilir.


