#!/usr/bin/env python
# coding: utf-8

# # 第三回の演習（回帰分析）

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model
import math
from mpl_toolkits.mplot3d import Axes3D  #3Dplot
import seaborn as sns


# ## 問題３.３

# In[2]:


df1 = pd.read_csv('lec03-LPGA.csv')


# In[3]:


plt.plot(df1['1R+2R+3R'],df1['4R'],'o')


# In[4]:


from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit(np.array(df1['1R+2R+3R']).reshape(-1,1), np.array(df1['4R']).reshape(-1,1)) # 予測モデルを作成

# 散布図
plt.scatter(np.array(df1['1R+2R+3R']).reshape(-1,1), np.array(df1['4R']).reshape(-1,1))

# 回帰直線
plt.title('Linear regression')
plt.plot(np.array(df1['1R+2R+3R']).reshape(-1,1), clf.predict(np.array(df1['1R+2R+3R']).reshape(-1,1)))
plt.xlabel('since third')
plt.ylabel('fourth_score')
plt.grid()
plt.show()

print("回帰係数= ", clf.coef_)
print("切片= ", clf.intercept_)
print("決定係数= ", clf.score(np.array(df1['1R+2R+3R']).reshape(-1,1), np.array(df1['4R']).reshape(-1,1)))


# ## 問題３.４

# In[5]:


list3_4 = np.array([9.4,10.1,16.9,22.1,24.6,26.6,32.7,32.5,26.6,23.0,17.7,12.1])
x1 = np.array([np.cos((2*math.pi*i)/12) for i in range(len(list3_4))])
x2 = np.array([np.sin((2*math.pi*i)/12) for i in range(len(list3_4))])
df2 = pd.DataFrame({ 'x1' : x1,'x2' : x2,'y' : list3_4})


# In[6]:


plt.plot(np.arange(12),x1)
plt.plot(np.arange(12),x2)
plt.plot(np.arange(12),list3_4/30)


# In[7]:


df2


# In[9]:


from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing

x = df2[['x1','x2']]
y = df2[['y']]
x1 = df2[['x1']]
x2 = df2[['x2']]
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


# In[10]:


y


# In[11]:


x


# In[11]:


# fig=plt.figure()
# ax=Axes3D(fig)

# ax.scatter3D(x1, x2, y)
# ax.set_xlabel("x1")
# ax.set_ylabel("x2")
# ax.set_zlabel("y")

# mesh_x1 = np.arange(x1.min()[0], x1.max()[0], (x1.max()[0]-x1.min()[0])/20)
# mesh_x2 = np.arange(x2.min()[0], x2.max()[0], (x2.max()[0]-x2.min()[0])/20)
# mesh_x1, mesh_x2 = np.meshgrid(mesh_x1, mesh_x2)
# mesh_y = model_lr.coef_[0][0] * mesh_x1 + model_lr.coef_[0][1] * mesh_x2 + model_lr.intercept_[0]
# ax.plot_wireframe(mesh_x1, mesh_x2, mesh_y)
# plt.show()


# ## シンプソンのパラドックス(問題3.5)

# In[12]:


df3 = pd.DataFrame({ 'x1' : [1,2,3,4,5,6],'x2' : [1,1,1,-1,-1,-1],'y' : [4,5,6,1,2,3]})


# In[15]:


##x1,x2の両方で説明する場合
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing

x = df3[['x1','x2']]
y = df3[['y']]
x1 = df3[['x1']]
x2 = df3[['x2']]
# sscaler = preprocessing.StandardScaler()
# sscaler.fit(x)
# sscaler.fit(y)
# x = sscaler.transform(x)
# y = sscaler.transform(y)

model_lr = LinearRegression()
model_lr.fit(x, y)

print("回帰係数= ",model_lr.coef_)
print("切片= ",model_lr.intercept_)
print("決定係数= ",model_lr.score(x, y))
###結果：y = x1 + 3*x2


# In[16]:


##x1だけで説明する場合
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing

x = df3[['x1','x2']]
y = df3[['y']]
x1 = df3[['x1']]
x2 = df3[['x2']]
# sscaler = preprocessing.StandardScaler()
# sscaler.fit(x)
# sscaler.fit(y)
# x = sscaler.transform(x)
# y = sscaler.transform(y)

model_lr = LinearRegression()
model_lr.fit(x1, y)

print("回帰係数= ",model_lr.coef_)
print("切片= ",model_lr.intercept_)
print("決定係数= ",model_lr.score(x1, y))
###結果：y = -0.54*x1 + 5.4


# In[ ]:




