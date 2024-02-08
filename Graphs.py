#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


stock_time = [0.0098 ,0.0012,0.0028,0.0010]
stock_acc = [94.1075,44.2918,71.3791,71.5147]


# In[3]:


intel_time = [0.0015,0.0012,0.0014,0.0013]
intel_acc =  [93.9858,44.2918,71.3774,71.5147]


# In[4]:


df = pd.DataFrame({
    'Model': ['RFC', 'LR', 'KNN','NB','SVM'],
    'Stock Time': [0.0098 ,0.0012,0.0028,0.0010,0.01],
    'Intel Time': [0.0015,0.0012,0.0014,0.0013,0.01]
})
df.plot(x="Model", y=["Stock Time", "Intel Time"], kind="bar")
plt.title("Stock vs Intel (Time)")
plt.show()


# In[5]:


df = pd.DataFrame({
    'Model': ['RFC', 'LR', 'KNN','NB'],
    'Stock Accuracy': [94.1075,44.2918,71.3791,71.5147],
    'Intel Accuracy': [93.9858,44.2918,71.3774,71.5147]
})
df.plot(x="Model", y=["Stock Accuracy", "Intel Accuracy"], kind="bar")
plt.title("Stock vs Intel (Accuracy)")


# In[6]:


res_intel = [i / j for i, j in zip(intel_acc, intel_time)]
res_intel


# In[7]:


res = [i / j for i, j in zip(intel_acc, intel_time)]
df = pd.DataFrame({
    'Models': ['RFC', 'LR', 'KNN','NB'],
    'Accuracy/Time': [62657.2, 36909.833333333336, 50983.85714285714, 55011.307692307695]
})
df.plot(x="Models", y=["Accuracy/Time"], kind="bar")
plt.title("Intel Model Score")


# In[8]:


res_stock = [i / j for i, j in zip(stock_acc, stock_time)]
res_stock


# In[9]:


df = pd.DataFrame({
    'Models': ['RFC', 'LR', 'KNN','NB'],
    'Accuracy/Time': [9602.80612244898, 36909.833333333336, 25492.535714285714, 71514.7]
})
df.plot(x="Models", y=["Accuracy/Time"], kind="bar")
plt.title("Stock Model Score")


# In[10]:


model_stock = [9602.80612244898, 36909.833333333336, 25492.535714285714, 71514.7]
model_intel = [62657.2, 36909.833333333336, 50983.85714285714, 55011.307692307695]
a = []
for i in range(0,len(model_stock)):
    if(model_stock[i]<=model_intel[i]):
            a.append("Intel")
    else:
            a.append("Stock")
a


# In[ ]:




