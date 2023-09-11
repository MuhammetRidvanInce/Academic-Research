#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
import scipy
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


# In[2]:


# Grafik çizimi için Fonksiyon
def plot_series(data_set):
  columns = data_set.columns

  fig, ax = plt.subplots(3,3, figsize = (12,12))
  index = 0
  for i in range(3):
    for j in range(3):
      ax[i][j].plot(data_set[columns[index]])
      ax[i][j].set_title(columns[index], fontweight = "bold")
      index +=1


# In[3]:


# Mevsimsellik ve trendden ayrıştırma ayrıştırılması için fonksiyon
def mevsim_ayristir(orijinal_veri_seti, donusturulen_veri_seti):
  columns = orijinal_veri_seti.columns
  for seri in columns:
    decompose = seasonal_decompose(orijinal_veri_seti[seri], model="multiplicative")
    donus_veri = decompose.observed / decompose.seasonal
    donusturulen_veri_seti[seri] = donus_veri
  return donusturulen_veri_seti


# In[4]:


# Durağanlık Testi İçin Fonksiyon
def stationary(x):
    result = adfuller(x)
    kv = result[0]
    p_value = result[1]
    lags = result[2]
    t_stats = result[4].values()
    print("-------------------")
    print("Kritik değer : {}\nP_Value : {}\nlags: {}\nt_stats: {}".format(kv,p_value,lags, t_stats))
    print("-------------------")


# In[5]:


raw_data = pd.read_csv("https://raw.githubusercontent.com/MutadisMutandis/datas2/main/D%C3%BCzce.csv").dropna()
raw_data
raw_data["index"] = pd.date_range('2016-01-01', periods=74, freq='M')

raw_data = raw_data.set_index('index')
raw_data = raw_data[["Elec_Consumption", "Price_TL", "Price_UDS","Exc_Rate", "Noc", "Price_Index"," House_Sell"," Weather_Forecast", "Working_Day"]] # Price_incex_2 veri setinden çıkarıldı, price_index'in % değişimi alınınca aynısını verecektir.
# Weather forecast ham verisindeki negatif iki değer min pozitif değere çevrildi.
raw_data.head()


# In[6]:


# Ham veriler grafik
plot_series(raw_data)


# In[7]:


# Mevsimsellik Aytıştırma
data_sets = pd.DataFrame(columns = raw_data.columns, index = raw_data.index)
raw_data_s = mevsim_ayristir(raw_data, data_sets)
raw_data_s.head()


# In[8]:


# Mevsimsellikten ayrıştırılan verilerin grafikleri
plot_series(raw_data_s)


# In[9]:


# Verilerin logaritmasının alınması
raw_data_s_l = np.log(raw_data_s)
raw_data_s_l.head()


# In[10]:


# Logaritmalı serilerin grafikleri
plot_series(raw_data_s_l)


# In[11]:


for col in raw_data_s_l.columns:
    print("\n{} Durağanlık Sonuçları".format(col))
    stationary(raw_data_s_l[col])
# Elec_Consumption , Price_TL , Price_UDS , Exc_Rate , Noc , Price_Index , Working_Day Durağan Değil fark alınması gerekiyor.


# In[12]:


# Fark alınmış seriler
raw_data_s_l_diff = raw_data_s_l.diff().dropna()
raw_data_s_l_diff.head()


# In[ ]:


#Fark alınmış serilerin grafikleri
plot_series(raw_data_s_l_diff)


# In[ ]:


# Tekrar durağanlık analizi
for col in raw_data_s_l_diff.columns:
    print("\n{} Durağanlık Sonuçları".format(col))
    stationary(raw_data_s_l_diff[col])
# Tüm serilerde durağanlık testi sonuçları durağanlığın sağlandığı yönünde.
# Kullanılacak seri seti raw_data_s_l_diff olacaktır.


# # Modelin Kurulması

# In[ ]:


# Eğitim ve test veri setinin oluşturulması
X = raw_data_s_l_diff.iloc[:,1:].values
y = raw_data_s_l_diff.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
np.shape(X_train)


# In[ ]:


model_Regresyon = LinearRegression().fit(X_train, y_train)
y_pred = model_Regresyon.predict(X_test)


# In[ ]:


X = np.append(arr = np.ones((73,1)).astype(int), values = X, axis = 1)
X_yeni = X[:, [0,1,2,3,4,5,6,7,8]]
np.shape(X_yeni)


# In[ ]:


model_resgresyon_OLS = sm.OLS(endog = y, exog = X_yeni).fit()
print(model_resgresyon_OLS.summary())


"""
X0: Constant             X
X1: Price_TL     
X2: Price_USD
X3: Exc_Rate
X4: Noc                  X
X5: Price_Index          X
X6: House_Sell 
X7: Weather_Forecast     X
X8: Working_Day          X
"""

# Anlamsız oldukları için Sabit, X4, X5, X7 ve X8 modelden çıkarıldı.


# In[ ]:


X_opt = X[:, [1,2,3,6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

"""
X1: Price_TL     
X2: Price_USD
X3: Exc_Rate
X4: House_Sell 
"""


# In[ ]:


import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import OLSInfluence


# In[ ]:


kalintilar = regressor_OLS.resid


# In[ ]:


fig, ax = plt.subplots(figsize = (10,4))
ax.bar(range(len(kalintilar)), kalintilar)
ax.scatter(range(len(kalintilar)), kalintilar)
ax.set_xlabel("Gözlem Değerleri")
ax.set_ylabel("Kalinti Değerleri")


# In[ ]:


kalintilar


# In[ ]:


X = X_opt
y = raw_data_s_l_diff.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
np.shape(X_train)


# In[ ]:


model_Regresyon = LinearRegression().fit(X_train, y_train)
model_Regresyon.coef_


# In[ ]:


fig, (ax1, ax2) = plt.subplots(2,1, figsize = (12,8))
ytrain_pred = []
for k in range(len(X_train)):
  ytrain_pred.append(np.sum((X_train*model_Regresyon.coef_)[k]))
ax1.plot(y_train,"-k", label = "Gerçek Değerler" )
ax1.plot(ytrain_pred, "--r", label = "Tahmin Değerleri")

ax1.set_xlabel("Değer")
ax1.set_ylabel("Gözlemler")
ax1.set_title("Eğitim Verisi için Gerçek Değer - Tahmin Değeri Karşılaştırma", fontsize = 15)
ax1.legend()



ytest_pred = []
for k in range(len(X_test)):
  ytest_pred.append(np.sum((X_test*model_Regresyon.coef_)[k]))
ax2.plot(y_test,"-k", label = "Gerçek Değerler" )
ax2.plot(ytest_pred, "--r", label = "Tahmin Değerleri")

ax2.set_xlabel("Değer")
ax2.set_ylabel("Gözlemler")
ax2.set_title("Test Verisi için Gerçek Değer - Tahmin Değeri Karşılaştırma", fontsize = 15)
ax2.legend()
plt.tight_layout()

