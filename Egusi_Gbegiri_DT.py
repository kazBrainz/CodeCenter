#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd

xoup = pd.read_csv('soupdtnew.csv')
#print(xoup.head())

print(xoup)


# In[8]:


#xoup.to_csv('soupdt.csv', index=False)


# In[9]:


X = xoup[['Melon_seed', 'Pumpkin_seed','Locust_beans', 
         'Crayfish_powder','Green_leaf_vege','Onions','Palm_oil',
          'Dried_pepper','Brown_beans','Black_eyed_peas']]
y = xoup.Soup_egusi


# In[10]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X,y)


# In[11]:


from dtreeviz.trees import dtreeviz
viz = dtreeviz(dt,X,y,
              target_name='Is_Soup_egusi?',
              feature_names=X.columns,
              class_names=["No","Yes"],
              scale=2.0)
viz.view()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




