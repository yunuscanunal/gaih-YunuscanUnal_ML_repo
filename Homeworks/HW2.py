#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression


# In[40]:


# Şeker Hastalığı Veri Kümesini içe aktarın
from sklearn.datasets import load_diabetes
Xb, yb = load_diabetes(return_X_y=True)

df_diabetes = pd.DataFrame(Xb, columns= load_diabetes().feature_names)
df_diabetes


# In[41]:


df_diabetes.info()


# In[42]:


df_diabetes.describe().T


# In[43]:


# Yinelenen değerleri ve eksik verileri kontrol edin
print("Sum of Duplicated Data: ",df_diabetes.duplicated().sum())
print("Sum of NA Data:\n",df_diabetes.isna().sum())
# There is no duplicated or NA data in dataset.


# In[44]:


# Her özellik için verileri görselleştirin (pairplot, distplot)
sns.pairplot(df_diabetes)


# In[45]:


# Korelasyon matrisini bastırın ve yorumlayın
corr = df_diabetes.corr()
corr


# In[46]:


plt.figure(figsize=(15,15))
sns.heatmap(corr, cmap="BuPu")


# # Korelasyon yorumu:
# ## Pozitif Corr 
# - Verimizin s2 özelliği(feature) ile s1 özelliği pozitif, yüksek korelasyona, ilişkiye  sahiptir. Bu da bize s1 değerimiz artarken s2 değerimiz de aynı anda benzer şekilde arttığını, s1 azalırken de aynı şekilde s2nin de azaldığını gösterir.
# - Aynı şekilde s2 özelliğimiz ile s4 özelliğimiz de pozitif, yüksek korelasyona sahiptir.
# - s4 ile s5 de pozitif, yüksek korelasyona sahiptir.
# 
# ## Negatif Corr
# - Verimizin s3 özelliği(feature) ile s4 özelliği negatif, yüksek korelasyona, ilişkiye sahiptir. Bu da bize s3 değerimiz artarken s4 değerimiz de aynı anda benzer oranda azaldığını, s3 azalırken de aynı şekilde s4nin arttığını gösterir. 
# 
# ## Zero Corr
# - s3 özelliğimiz ile s1 özelliğimiz arasındaki korelasyon değerimiz: 0.051519. Bu da s1 ile s3 arasında herhangi korelasyonun (ilişkinin), yada benzerliğin olmadığı anlamına gelebilir.

# In[47]:


# İlişkili bulduğunuz özellikleri eleyin (korelasyon matrisini kontrol ederek)
df_diabetes = df_diabetes.drop(["s2","s4"] ,axis=1).reset_index(drop= True)
df_diabetes


# In[48]:


# Aykırı değerleri işleyin (IsolationForest kullanabilirsiniz)
from scipy import stats
z = np.abs(stats.zscore(df_diabetes))
z


# In[49]:


print("length of anomaly list: ", len(np.where(z>3)[0]))
print("Index of Anomaly data:", np.where(z>3)[0])


# In[51]:


outliers = list(set(np.where(z>3)[0]))
new_df = df_diabetes.drop(outliers, axis=0).reset_index(drop= False)
display(new_df)

y_new = yb[list(new_df["index"])]
len(y_new)


# In[52]:


new_df = new_df.drop(["index"] ,axis=1).reset_index(drop= True)
new_df


# In[53]:


# Özellikleri ölçekleyin. (scaling)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X_scaled = StandardScaler().fit_transform(new_df)
X_scaled


# ## Without Outliers (Outliers dropped.)

# In[54]:


# Veri kümesini eğitim ve test verisetlerine ayırın.
from sklearn.model_selection import train_test_split, cross_validate

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_new, test_size=0.3, random_state=22)


# In[55]:


# Lasso ve Rdige modellerini Sklearn'dan içe aktarın.
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score


# In[96]:


# Ridge için 5 farklı alfa değeri tanımlayın ve modelleri eğitin. R^2 değerlerini 
# hem eğitim hem de test verisetleri için yazdırın.

for i in (1e-8, 1e-4, 1e-3, 1e-2, 1):
    ridge_model = Ridge(alpha=i)
    ridge_model.fit(X_train, y_train)
    print("Simple Train: ", ridge_model.score(X_train,y_train))
    print("Simple Test: ", ridge_model.score(X_test,y_test))
    print("-"*30)
    print("r2score Train: ", r2_score(y_train,ridge_model.predict(X_train)))
    print("r2score test: ", r2_score(y_test,ridge_model.predict(X_test)))
    print("*"*30)
    print("*"*30,"\n","\n")


# In[97]:


# Lasso için 5 farklı alfa değeri tanımlayın ve modelleri eğitin. R^2 değerlerini 
# hem eğitim hem de test verisetleri için yazdırın.

for i in (1e-8, 1e-4, 1e-3, 1e-2, 1):
    lasso_model = Lasso(alpha=i)
    lasso_model.fit(X_train, y_train)
    print("Simple Train: ", lasso_model.score(X_train,y_train))
    print("Simple Test: ", lasso_model.score(X_test,y_test))
    print("-"*30)
    print("r2score Train: ", r2_score(y_train,lasso_model.predict(X_train)))
    print("r2score Test: ", r2_score(y_test,lasso_model.predict(X_test)))
    print("*"*30)
    print("*"*30,"\n","\n")


# ## With Outliers (Outliers did not drop.)

# In[58]:


df_diabetes


# In[63]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(df_diabetes, yb, test_size=0.3, random_state=22)


# In[87]:


#RIDGE

for i in (1e-8, 1e-4, 1e-3, 1e-2, 1):
    ridge_model2 = Ridge(alpha=i)
    ridge_model2.fit(X_train2, y_train2) 
    print("Simple Train: ", ridge_model2.score(X_train2,y_train2))
    print("Simple Test: ", ridge_model2.score(X_test2,y_test2))
    print("-"*30)
    print("r2score_train: ", r2_score(y_train2,ridge_model2.predict(X_train2)))
    print("r2score_test: ", r2_score(y_test2,ridge_model2.predict(X_test2)))
    print("*"*30)
    print("*"*30,"\n","\n")


# In[88]:


#LASSO

for i in (1e-8, 1e-4, 1e-3, 1e-2, 1):
    lasso_model2 = Lasso(alpha=i)
    lasso_model2.fit(X_train2, y_train2)
    print("Simple Train: ", lasso_model2.score(X_train2,y_train2))
    print("Simple Test: ", lasso_model2.score(X_test2,y_test2))
    print("-"*30)
    print("r2score_train: ", r2_score(y_train2,lasso_model2.predict(X_train2)))
    print("r2score_test: ", r2_score(y_test2,lasso_model2.predict(X_test2)))
    print("*"*30)
    print("*"*30,"\n","\n")


# ###### Sonuçlar hakkında yorum yapın. En iyi modelin katsayısını yazdırın.
# 
# - model = ayrık değerler atıldı.
# - model2 = ayrık değerler kaldı.
# 
# - Modellerimizin sonuçlarını kıyaslarsak, modelimiz aykırı değerleri atmadığımız versiyonda (model2) hem model.score() değerleri hem de r^2 değerleri için daha yüksek başarı değeri verdi. O yüzden model2 yi yorumlamaya devam edeceğim.
# 
# - Model2 için Lasso ve Ridge alfa değerlerimiz (10^-8, 10^-4, 10^-3, 10^-2, 1) arasındaki en başarılı modeller:
# - Ridge için alfa değerimiz = 10^-2, modelimizin test kümesinde r2 başarısı = 0.514
# - Lasso için alfa değerimiz = 10^-2, modelimizin test kümesinde r2 başarısı = 0.515
# - r2 score() için en başarılı modelimiz Lasso, alfa değerimiz 0.01
# - Genel olarak başarısız diye nitelendirebileğimiz bir model eğitmiş olduk.

# In[120]:


#katsayı
lasso_model2 = Lasso(alpha=0.01)
lasso_model2.fit(X_train2, y_train2)
print(f"Lasso Model2 Coef_ : {lasso_model2.coef_}\n\n")
print(f"Lasso Model2 Intercept_ : {lasso_model2.intercept_}")

