#!/usr/bin/env python
# coding: utf-8

# In[12]:


# TRAINING SCRIPT
# Import and parameter section
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import glob
from imblearn.over_sampling import RandomOverSampler
import glob
import pickle


# In[16]:


file_name = glob.glob('../data/*.csv')
df = pd.read_csv(file_name[0],  sep = ",")
y = df.pop('target')
x = df.copy()


# In[3]:


ros = RandomOverSampler(random_state=0)
x, y = ros.fit_resample(x, y)


# In[5]:


# cat vars
#x = pd.get_dummies(df, columns=['loan_type','customer_postal']) # all cats
# scale
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
#PCA
#pca = PCA(n_components=30)
#x = pca.fit_transform(x)


# In[6]:


# Divide data into test and training
test_size = 0.25
seed = 7
X_train, X_test, y_train, y_test =  train_test_split(x, y, test_size=test_size, random_state=seed)


# In[ ]:


# # Grid secrh

# param_grid = {'bootstrap': [True, False],
#  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#  'max_features': ['auto', 'sqrt'],
#  'min_samples_leaf': [1, 2, 4],
#  'min_samples_split': [2, 5, 10],
#  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

# gsc = GridSearchCV(
#         estimator=RandomForestClassifier(),
#         param_grid=param_grid,
#         cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

# gsc.fit(X_train, y_train)


# In[ ]:


#/print(gsc.best_params_)


# In[7]:


# Select algorithm
model = RandomForestClassifier(bootstrap=True, max_depth=80, max_features='sqrt',
                               min_samples_leaf=1, min_samples_split=5, 
                               n_estimators=1200, n_jobs= -1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
predictions


# In[8]:


print(accuracy_score(y_test, predictions))


# In[9]:


df_cm = confusion_matrix(y_test, predictions)
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True)


# In[10]:



pickle.dump(model, open('../models/model.pickle', 'wb'))

