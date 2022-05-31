#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

soup = pd.read_csv('soup_decision.csv')
print(soup)


# In[4]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[5]:


soup = pd.read_csv('soup_decision.csv')
print(soup.head())


# In[6]:


print('Dataset length: ', len(soup) )


# In[7]:


print('Dataset shape: ', soup.shape )


# In[8]:


X = soup.values[:, 1:11]
y = soup.values[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 100)

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,max_depth=3, min_samples_leaf=11)
clf_entropy.fit(X_train, y_train)


# In[36]:


pred = np.array([4, 0, 1, 2, 1, 0, 0.5, 0.0, 0, 0])

y_pred_ent = clf_entropy.predict(pred.reshape(1,-1))
print(y_pred_ent)


# In[ ]:




