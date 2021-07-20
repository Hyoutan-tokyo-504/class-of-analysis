#!/usr/bin/env python
# coding: utf-8

# # モンテカルロ法の実装

# In[2]:


import numpy as np
import random
import statistics
import collections
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 問題２.５

# In[2]:


def function(a,b,c,d,e):
    return 1/(a+(1/(b+1/(c+1/(d+1/e)))))


# In[3]:


def MonteCarlo(n):
    Montelist = []
    for i in range(n):
        randomlist = 1 + np.random.rand(5)
        Montelist.append(function(randomlist[0],randomlist[1],randomlist[2],randomlist[3],randomlist[4]))
    montemean = np.mean(Montelist)
    monteerror = np.std(Montelist)/np.sqrt(n)
    return montemean, monteerror


# In[4]:


MonteCarlo(10000)


# ## 問題２.６

# In[5]:


#問題2.6
##最大頻度の階級が４である確率
def MonteCarlo2_miss(n):
    Montelist = []
    for i in range(n):
        modedic = collections.Counter(random.choices(np.arange(1,7), k=10)).most_common()
        if modedic[0][0] == 4:
                Montelist.append(1)
        else:
            for i in range(1,len(modedic)):
                if modedic[i][0] == 4:
                    if modedic[i][1] == modedic[0][1]:
                        Montelist.append(1)
                        break
                    else:
                        Montelist.append(0)
                        break
                else:
                    Montelist.append(0)
                    break
            
    return np.mean(Montelist), len(Montelist)


# In[6]:


MonteCarlo2_miss(1000)


# In[72]:


def MonteCarlo2(n):
    Montelist = []
    for i in range(n):
        modedic = collections.Counter(random.choices(np.arange(1,7), k=10)).most_common()
        if modedic[0][1] == 4:
            Montelist.append(1)
        else:
            Montelist.append(0)
    mean = np.mean(Montelist)
    stderror = np.sqrt((mean*(1-mean))/n)
    return mean, stderror


# In[73]:


MonteCarlo2(10000)


# ## 確率と標準誤差の収束を見る

# In[74]:


start = time.time()
Montelist = []
for i in range(1,10001,100):
    Montelist.append(MonteCarlo2(i))
end = time.time() - start


# In[75]:


end


# In[76]:


plt.plot(np.arange(1,101),Montelist)
##確率（青）の収束と標準誤差（オレンジ）の収束


# In[ ]:




