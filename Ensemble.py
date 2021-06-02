#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[14]:


test1 = pd.read_csv("test_sample_74.csv")
test2 = pd.read_csv("test_sample_50epoch.csv")
test3 = pd.read_csv("test_sample_100epoch.csv")
result = pd.DataFrame(columns=["Id","Category"])
result.iloc[:,0]=test1.iloc[:,0]


# In[20]:


for i in range(len(test1.iloc[:,1])):
    count1 = 0
    count2 = 0
    count3 = 0
    if test1.iloc[i,1] == 1:
        count1+=1
    elif test1.iloc[i,1]==2:
        count2+=1
    else:
        count3+=1
    if test2.iloc[i,1] == 1:
        count1+=1
    elif test2.iloc[i,1]==2:
        count2+=1
    else:
        count3+=1
    if test3.iloc[i,1] == 1:
        count1+=1
    elif test3.iloc[i,1]==2:
        count2+=1
    else:
        count3+=1
        
    if count1 > count2 and count1 > count3:
        result.iloc[i,1]=1
    elif count2 > count1 and count2 > count3:
        result.iloc[i,1]=2
    else:
        result.iloc[i,1]=3


# In[26]:



result.to_csv("result.csv",index = False)


# In[94]:




