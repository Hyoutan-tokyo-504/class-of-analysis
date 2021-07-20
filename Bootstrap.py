#!/usr/bin/env python
# coding: utf-8

# # ブートストラップ法の実装

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats


# In[2]:


data = np.array([-1, 3, -1, 0, 2, -1, 3, 2, 1, 1, 2, 1, 3, 0, 2])


# In[3]:


error1 = np.std(data)/np.sqrt(len(data))


# In[4]:


error1


# In[5]:


mean_data = np.average(data)
sem_data = np.std(data)/np.sqrt(len(data))
median_data = np.median(data)


# In[1]:


#ブートストラップ法による中央値と平均の標準誤差
#平均の標準誤差はそれぞれの平均の標準偏差で求められることを利用している
def bootstrap(x,repeats):
    # placeholder (column1: mean, column2: median)
    vec = np.zeros((3,repeats))
    for i in np.arange(repeats):
        # resample data with replacement
        re = np.random.choice(len(x),len(x),replace=True)
        re_x = x[re]

        # compute mean and median of the "new" dataset
        vec[0,i] = np.mean(re_x)
        vec[1,i] = np.median(re_x)
        vec[2,i] = scipy.stats.skew(re_x)

    # histogram of median from resampled datasets
    sns.distplot(vec[2,:], kde=False)

    # compute bootstrapped standard error of the mean,
    # and standard error of the median
    b_mean_sem = np.std(vec[0,:])
    b_median_sem = np.std(vec[1,:])
    b_skew_sem = np.std(vec[2,:])

    return b_median_sem, b_mean_sem, b_skew_sem


# In[7]:


bootstrap(data,1000)


# In[ ]:




