#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[2]:


df1 = pd.read_csv("bengaluru_house_prices.csv")
df1.head()


# In[3]:


df1['area_type'].unique()


# In[4]:


df1['area_type'].value_counts()


# In[5]:


df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.shape


# In[6]:


df2.isnull().sum()


# In[7]:


df2.shape


# In[8]:


df3 = df2.dropna()
df3.isnull().sum()


# In[9]:


df3.shape


# In[10]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# In[11]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[12]:


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head(2)


# In[13]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# In[14]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats


# In[15]:


location_stats.values.sum()


# In[16]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[17]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[18]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[19]:


df5.shape


# In[20]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[21]:


df6.price_per_sqft.describe()


# In[22]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# In[23]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[24]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[25]:


df9.head()


# In[26]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[27]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[29]:


X = df11.drop(['price'],axis='columns')
X.head()


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[ ]:




