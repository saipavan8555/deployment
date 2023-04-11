#!/usr/bin/env python
# coding: utf-8

# In[5]:import required libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


print('Importing Dependencies.....')

import pickle

train=pd.read_csv('house_rent.csv')
# In[2]:read the data

#print('Reading the data......')
print(train.head(5))


# In[3]:split the dataset into dependent and independent variables x=? and y= ?
x=train.drop('rent',axis=1)
y=train['rent']

cols=['city','type','bhk']
x[cols]=x[cols].astype('category')
cat_cols=x.select_dtypes(include=['category']).columns

x=pd.get_dummies(columns=cat_cols,data=x,prefix=cat_cols,prefix_sep='_',drop_first=True)

print(x)

# In[6]:build the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
print('Building Model...........')



# In[7]:fit the model
model=lm.fit(x,y)



# In[8]:save the model as pickel file


print('saving model as pkl file.......')
pickle.dump(model, open('model.pkl','wb'))


