#!/usr/bin/env python
# coding: utf-8

# # 問題１（モンテカルロ法）

# In[27]:


import numpy as np
import random
import statistics
import collections
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


#外部から持ってきた関数
def has_duplicates(seq):
    return len(seq) != len(set(seq))


# In[29]:


def Trampgame(n):
    scorelist = []
    #試行回数分繰り返す
    for i in range(n):
        #トランプリストの生成
        list1 = list(range(1,14))*4
        #ゲーム開始
        for j in range(10):
            chosenlist = []
            #要素の抽出と要素の削除
            for k in range(5):
                number = random.choice(list1)
                chosenlist.append(number)
                list1.remove(number)
            #重複の確認
            if has_duplicates(chosenlist) == True:
                scorelist.append(j+1)
                break
            else:
                if j == 9:
                    scorelist.append(20)
                    break
    scorelist = np.array(scorelist)
    mean = np.mean(scorelist)
    error = np.std(scorelist)/np.sqrt(n)
    return [mean, error]
# ,collections.Counter(scorelist).most_common()


# ## 結果の確認と収束の様子

# In[30]:


Trampgame(10000)


# In[31]:


Resultlist = []
for i in range(1,10001,100):
    Resultlist.append(Trampgame(i))
result = np.array(Resultlist) - np.array([2,0])
# plt.xlim(80, 100)
plt.ylim(-0.1, 0.2)
plt.plot(np.arange(1,101)*100,result)


# # 問題２（重回帰分析）

# In[147]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model
import math
from mpl_toolkits.mplot3d import Axes3D  #3Dplot
import seaborn as sns
import statsmodels.api as sm


# ## データ成形

# In[148]:


def direction_dummy(value):
    directionlist = ['北','北北東','北東','東北東','東','東南東','南東','南南東','南','南南西','南西','西南西','西','西北西','北西','北北西']
    arglist = [-np.cos(i*math.pi/8) for i in range(len(directionlist))]
    return arglist[directionlist.index(value)]


# In[149]:


def direction_dummy2(value):
    directionlist = ['北','北北東','北東','東北東','東','東南東','南東','南南東','南','南南西','南西','西南西','西','西北西','北西','北北西']
    arglist = [-np.sin(i*math.pi/8) for i in range(len(directionlist))]
    return arglist[directionlist.index(value)]


# In[150]:


direction_dummy('北北東')


# ## 東京のデータ

# In[177]:


df = pd.read_csv('temp.csv',encoding='shift_jis')
df = df.reset_index()
df = df.drop(df.index[[0,1,2]])
df = df.drop(df.columns[[2,3,4,6,7,9,10,12,13,15,16]], axis=1)
df = df.rename(columns={'level_0': 'month', 'level_1': 'suntime', 
                        'level_5': 'wind_verocity', 
                        'level_8': 'wind_direction', 
                        'level_11': 'cloud', 'level_14': 'temp'})
df = df.reset_index()
df = df.drop(df.columns[[0,1]], axis=1)
df['month'] = [-np.cos(i*math.pi/6) for i in range(len(df['suntime']))]
df['wind_directdummy'] = [direction_dummy(value) for value in df['wind_direction']]
df['wind_directdummy2'] = [direction_dummy2(value) for value in df['wind_direction']]
df = df.drop('wind_direction',axis=1)
for column in df.columns:
    df[column] = [float(value) for value in df[column]]


# In[178]:


df.head()


# In[179]:


from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing

X = df[['wind_verocity','cloud','month','wind_directdummy','wind_directdummy2','suntime']]
# X = X.apply(lambda x: (x-x.mean())/x.std(), axis=0)
# x = df[['cloud','month','wind_directdummy']]
Y = df[['temp']]
# Y = Y.apply(lambda y: (y-y.mean())/y.std(), axis=0)

model_lr = LinearRegression()
model_lr.fit(X, Y)
print("回帰係数= ",model_lr.coef_)
print("切片= ",model_lr.intercept_)
print("決定係数= ",model_lr.score(X, Y))


# In[171]:


model_lr.coef_[0][1]


# In[172]:


X.head()


# In[ ]:





# ### 他のモデルでの検証

# In[180]:


X = df[['wind_verocity','cloud','month','wind_directdummy','wind_directdummy2','suntime']]
X = X.apply(lambda x: (x-x.mean())/x.std(), axis=0)
Y = df[['temp']]
Y = Y.apply(lambda y: (y-y.mean())/y.std(), axis=0)
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())


# In[181]:


X.head()


# ### 可視化

# In[182]:


plt.figure(figsize=(20,10))
plt.subplot(231)
plt.plot(np.array([float(value) for value in df['temp']]), label="temperature")
plt.plot(np.array([float(value) for value in df['wind_verocity']]), label="verocity")
plt.legend()
plt.subplot(232)
plt.plot(np.array([float(value) for value in df['temp']]), label="temperature")
plt.plot(np.array([float(value) for value in df['cloud']]), label="cloud")
plt.legend()
plt.subplot(233)
plt.plot(np.array([float(value) for value in df['temp']]), label="temperature")
plt.plot(np.array([float(value) for value in df['month']]), label="month")
plt.legend()
plt.subplot(234)
plt.plot(np.array([float(value) for value in df['temp']]), label="temperature")
plt.plot(np.array([float(value) for value in df['wind_directdummy']]), label="wind_ns")
plt.legend()
plt.subplot(235)
plt.plot(np.array([float(value) for value in df['temp']]), label="temperature")
plt.plot(np.array([float(value) for value in df['wind_directdummy2']]), label="wind_ew")
plt.legend()
plt.subplot(236)
plt.plot(np.array([float(value) for value in df['temp']]), label="temperature")
plt.plot(np.array([float(value) for value in df['suntime']]), label="suntime")
plt.legend()
plt.show()


# In[183]:


s1=pd.Series(df['temp'])
s2=pd.Series(df['suntime'])
res=s1.corr(s2)


# In[184]:


res


# ### ブートストラップ法

# In[192]:


# 同じ東京データ
df = pd.read_csv('temp.csv',encoding='shift_jis')
df = df.reset_index()
df = df.drop(df.index[[0,1,2]])
df = df.drop(df.columns[[2,3,4,6,7,9,10,12,13,15,16]], axis=1)
df = df.rename(columns={'level_0': 'month', 'level_1': 'suntime', 
                        'level_5': 'wind_verocity', 
                        'level_8': 'wind_direction', 
                        'level_11': 'cloud', 'level_14': 'temp'})
df = df.reset_index()
df = df.drop(df.columns[[0,1]], axis=1)
df['month'] = [-np.cos(i*math.pi/6) for i in range(len(df['suntime']))]
df['wind_directdummy'] = [direction_dummy(value) for value in df['wind_direction']]
df['wind_directdummy2'] = [direction_dummy2(value) for value in df['wind_direction']]
df = df.drop('wind_direction',axis=1)
for column in df.columns:
    df[column] = [float(value) for value in df[column]]


# In[193]:


#各回帰係数に関するランダム性を持たせるための標準偏差生成関数
def this_standardspawn(df):
    standardlist = []
    for column in df.columns:
        matrix = np.zeros((12,int(len(df[column])/12)))
        for i in range(12):
            for j in range(i,len(df[column]),12):
                matrix[i,int(j/12)] = df[column][j]
        averagelist = [np.std(matrix[i]) for i in range(len(matrix))]
        standardlist.append(np.mean(averagelist))
    return standardlist


# In[194]:


##関数を更新する関数
def df_update(df):
    stdlist = this_standardspawn(df)
    col_number = 0
    for column in df.columns:
        df[column] = [df[column][i] + np.random.normal(loc=0.0, scale=stdlist[col_number], size=None) for i in range(len(df[column]))]
        col_number += 1
    return df


# In[195]:


#回帰係数のリストを作る
coeflist = []
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing
for i in range(1000):
    X = df[['wind_verocity','cloud','month','wind_directdummy','wind_directdummy2','suntime']]
    X = X.apply(lambda x: (x-x.mean())/x.std(), axis=0)
    X = df_update(X)
    Y = df[['temp']]
    Y = Y.apply(lambda y: (y-y.mean())/y.std(), axis=0)
    model_lr = LinearRegression()
    model_lr.fit(X, Y)
    coeflist.append(model_lr.coef_[0])
#回帰係数の標準誤差を求める
coeflist = np.array(coeflist)
stderrorlist = [np.std(coeflist[:,i]) for i in range(len(coeflist[0]))]  
stderrorlist


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 地元富山のデータ

# In[185]:


df = pd.read_csv('toyama.csv',encoding='shift_jis')
df = df.reset_index()
df = df.drop(df.index[[0,1,2]])
df = df.drop(df.columns[[2,3,4,6,7,9,10,12,13,15,16]], axis=1)
df = df.rename(columns={'level_0': 'month', 'level_1': 'suntime', 
                        'level_5': 'wind_verocity', 
                        'level_8': 'wind_direction', 
                        'level_11': 'cloud', 'level_14': 'temp'})
df = df.reset_index()
df = df.drop(df.columns[[0,1]], axis=1)
df['month'] = [-np.cos(i*math.pi/6) for i in range(len(df['suntime']))]
df['wind_directdummy'] = [direction_dummy(value) for value in df['wind_direction']]
df['wind_directdummy2'] = [direction_dummy2(value) for value in df['wind_direction']]
df = df.drop('wind_direction',axis=1)
for column in df.columns:
    df[column] = [float(value) for value in df[column]]


# In[186]:


from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing

x = df[['wind_verocity','cloud','month','wind_directdummy','wind_directdummy2','suntime']]
y = df[['temp']]

model_lr = LinearRegression()
model_lr.fit(x, y)

print("回帰係数= ",model_lr.coef_)
print("切片= ",model_lr.intercept_)
print("決定係数= ",model_lr.score(x, y))


# In[187]:


s1=pd.Series(df['temp'])
s2=pd.Series(df['wind_directdummy2'])
res=s1.corr(s2)
res


# In[188]:


plt.figure(figsize=(20,10))
plt.subplot(231)
plt.plot(np.array([float(value) for value in df['temp']]), label="temperature")
plt.plot(np.array([float(value) for value in df['cloud']]), label="cloud")
plt.legend()
plt.subplot(232)
plt.plot(np.array([float(value) for value in df['temp']]), label="temperature")
plt.plot(np.array([float(value) for value in df['wind_verocity']]), label="verocity")
plt.legend()
plt.subplot(233)
plt.plot(np.array([float(value) for value in df['temp']]), label="temperature")
plt.plot(np.array([float(value) for value in df['month']]), label="month")
plt.legend()
plt.subplot(234)
plt.plot(np.array([float(value) for value in df['temp']]), label="temperature")
plt.plot(np.array([float(value) for value in df['wind_directdummy']]), label="wind_ns")
plt.legend()
plt.subplot(235)
plt.plot(np.array([float(value) for value in df['temp']]), label="temperature")
plt.plot(np.array([float(value) for value in df['wind_directdummy2']]), label="wind_ew")
plt.legend()
plt.subplot(236)
plt.plot(np.array([float(value) for value in df['temp']]), label="temperature")
plt.plot(np.array([float(value) for value in df['suntime']]), label="suntime")
plt.legend()
plt.show()


# In[189]:


s1=pd.Series(df['temp'])
s2=pd.Series(df['suntime'])
res=s1.corr(s2)
res


# In[191]:


X = df[['wind_verocity','cloud','month','wind_directdummy','wind_directdummy2','suntime']]
X = X.apply(lambda x: (x-x.mean())/x.std(), axis=0)
Y = df[['temp']]
Y = Y.apply(lambda y: (y-y.mean())/y.std(), axis=0)
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())


# ## 過去60年間の東京都１２月のデータ解析

# In[68]:


df = pd.read_csv('december.csv',encoding='shift_jis')
df = df.reset_index()
df = df.drop(df.index[[0,1,2]])
df = df.drop(df.columns[[2,3,4,6,7,9,10,12,13,15,16,18,19,20,22,23,24,26,27,29,30,32,33]], axis=1)
df = df.rename(columns={'level_0': 'month', 'level_1': 'suntime', 
                        'level_5': 'wind_verocity', 'level_8': 'wind_direction', 
                        'level_11': 'cloud', 'level_14': 'temp', 'level_17': 'snow',
                        'level_21': 'rain', 'level_25': 'pressure','level_28': 'evaporate','level_31': 'fog'})
# df = df.reset_index()
# df = df.drop(df.columns[[0,1]], axis=1)
# df['month'] = [-np.cos(i*math.pi/6) for i in range(len(df['suntime']))]
# df['wind_directdummy'] = [direction_dummy(value) for value in df['wind_direction']]
# df['wind_directdummy2'] = [direction_dummy2(value) for value in df['wind_direction']]


# In[69]:


df.head()


# In[70]:


from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing

x = df[['suntime','wind_verocity','cloud','snow','rain','pressure','evaporate','fog']]
# x = df[['cloud']]
y = df[['temp']]
sscaler = preprocessing.StandardScaler()
sscaler.fit(x)
sscaler.fit(y)
x = sscaler.transform(x)
y = sscaler.transform(y)

model_lr = LinearRegression()
model_lr.fit(x, y)

print("回帰係数= ",model_lr.coef_)
print("切片= ",model_lr.intercept_)
print("決定係数= ",model_lr.score(x, y))


# In[2]:


#正則行列じゃないとエラー発生
import numpy as np
mat = [[1,0],[0,1]]
np.linalg.inv(mat)


# In[ ]:




