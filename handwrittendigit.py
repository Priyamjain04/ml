#!/usr/bin/env python
# coding: utf-8

# In[33]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[34]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[35]:


len(X_train)


# In[36]:


len(X_test)


# In[37]:


X_train[0]


# In[38]:


plt.matshow(X_train[0])


# In[39]:


X_train = X_train / 255
X_test = X_test / 255


# In[40]:


X_train[0]


# In[41]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[42]:


X_train_flattened.shape


# In[43]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[44]:


model.evaluate(X_test_flattened, y_test)


# In[45]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# In[46]:


plt.matshow(X_test[0])


# In[47]:


np.argmax(y_predicted[0])


# In[48]:


model.evaluate(X_test,y_test)


# In[ ]:




