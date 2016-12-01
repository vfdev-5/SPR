# # NN tryouts on SPR data, inspired by Kaggle Forum "When less is more"

# 
# - Load m month of data
# - Minimal data cleaning
# - Feature engineering
# 
# - Setup model
# 
# TRAIN : 201505, 201506
# 
# - Select only users that added products in 201506 comparing to 201505
# - FEATURES <- get_profile(ALL_FEATURES) : Select some profiles
# 
# [FEATURES|TARGETS]
# 
# X_train = [FEATURES] of the training part
# Y_train = [TARGETS]  of the training part
# 
# X_val = [FEATURES] of the validation part
# Y_val = [TARGETS]  of the validation part
# 
# TEST :
# 201606
# - All users
# [FEATURES]
# X_test = [FEATURES]

import os
import pandas as pd

import logging
logging.getLogger().handlers = []
logging.basicConfig(level=logging.DEBUG)

from common import load_data2, minimal_clean_data_inplace, preprocess_data_inplace, TARGET_LABELS


# In[3]:

TRAIN_FILE_PATH = os.path.join("..", "data", "train_ver2.csv")
TEST_FILE_PATH = os.path.join("..", "data", "test_ver2.csv")


# In[4]:

yearmonth_list = [201505, 201506]


# In[5]:

nb_months = len(yearmonth_list)


# In[6]:

nb_clients = 10000


# In[7]:

data_df = load_data2(TRAIN_FILE_PATH, yearmonth_list, nb_clients)


# In[8]:

minimal_clean_data_inplace(data_df)


# In[9]:

print data_df.shape
#data_df.tail()


# Encode non-numerical columns 

# In[10]:

preprocess_data_inplace(data_df)


# In[11]:

print data_df.shape
data_df.tail()


# In[12]:

clients = data_df['ncodpers'].unique()
print len(clients), (data_df['ncodpers'].value_counts() == nb_months).sum()
assert len(clients) == (data_df['ncodpers'].value_counts() == nb_months).sum()


# In[13]:

features = [
    u'ind_empleado', u'pais_residencia',
    u'sexo', u'age', u'ind_nuevo', u'antiguedad', u'indrel',
    u'ult_fec_cli_1t', u'indrel_1mes', u'tiprel_1mes', u'indresi',
    u'indext', u'conyuemp', u'canal_entrada', u'indfall', u'nomprov',
    u'ind_actividad_cliente', u'renta', u'segmento'    
]


# In[14]:

months = data_df['fecha_dato'].unique()
print months


# In[15]:

clients = data_df['ncodpers'].unique()
print len(clients)


# In[16]:

ll = len(clients)
for m in months:
    l = len(data_df[data_df['fecha_dato'] == m]['ncodpers'].unique())
    assert l == ll, "Number of clients should be identical for all monthes. (%s, %s, %s)" % (m, l, ll)


# ### Define train_val dataset :
# - Select only clients that choose new products in 201506 comparing with 201505

# In[17]:

trainval_df = data_df.sort_values(['fecha_dato', 'ncodpers'])


# In[18]:

dates1 = months[:-1]
dates2 = months[1:]


# In[19]:

print dates1, dates2


# In[20]:

tmp_df = trainval_df[['fecha_dato','ncodpers']]
tmp_df.loc[:,'target'] = trainval_df[TARGET_LABELS].sum(axis=1)
v1 = tmp_df[tmp_df['fecha_dato'].isin(dates2)]['target'].values
v2 = tmp_df[tmp_df['fecha_dato'].isin(dates1)]['target'].values
ll = min(len(v1), len(v2))
indices = tmp_df.index[ll:]
trainval_df.loc[indices,'diff'] = pd.Series(v1 - v2, index=indices)
del tmp_df, v1, v2


# In[21]:

trainval_df.sort_values(['ncodpers', 'fecha_dato']).head(10)


# In[22]:

X = trainval_df[(trainval_df['fecha_dato'].isin(dates2)) & (trainval_df['diff'] > 0)][features]
Y = trainval_df[(trainval_df['fecha_dato'].isin(dates2)) & (trainval_df['diff'] > 0)][TARGET_LABELS]


# In[23]:

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X.values, Y.values, train_size=0.70)


# In[24]:

from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
X_val = StandardScaler().fit_transform(X_val)


# In[25]:

print X_train.shape, X_val.shape
print Y_train.shape, Y_val.shape


# Setup NN model

# Keras model :
# 
# Sequential
# - Dense
# - Activation
# - Dropout

# In[26]:

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils


# Setup model 1

# In[ ]:

model = Sequential()
model.add(Dense(43, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(Y_train.shape[1], activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:

model.fit(X_train, Y_train, nb_epoch=1500, batch_size=10000, verbose=2)


# In[ ]:

# summarize performance of the model
scores = model.evaluate(X_val, Y_val, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))


# Setup model 2

# In[ ]:

model = Sequential()
model.add(Dense(43, init='uniform', input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(Y_train.shape[1], activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# In[ ]:

model.fit(X_train, Y_train, nb_epoch=1000, batch_size=10000, verbose=2)


# In[ ]:

# summarize performance of the model
scores = model.evaluate(X_val, Y_val, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))


# Setup model 3

# In[ ]:

model = Sequential()
model.add(Dense(50, init='uniform', input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(Y_train.shape[1], activation='softmax'))
model.compile(loss='mae', optimizer='nadam', metrics=['accuracy'])


# In[ ]:

model.fit(X_train, Y_train, nb_epoch=1000, batch_size=10000, verbose=2)


# In[ ]:

# summarize performance of the model
scores = model.evaluate(X_val, Y_val, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:




# ## Prediction part

# In[ ]:

#del X, Y, X_train, X_val, Y_train, Y_val, trainval_df, data_df


# Load the last month from the training dataset to get user last choice 

# In[ ]:

yearmonth_list = [201605]


# In[ ]:

lastmonth_df = load_data2(TRAIN_FILE_PATH, yearmonth_list)


# In[ ]:

minimal_clean_data_inplace(lastmonth_df)


# In[ ]:

clients_last_choice = lastmonth_df[['ncodpers'] + TARGET_LABELS]


# In[ ]:

test_df = load_data2(TEST_FILE_PATH, [])


# In[ ]:

test_df.head()


# In[ ]:

minimal_clean_data_inplace(test_df)


# In[ ]:

print test_df.shape
test_df.tail()


# Encode non-numerical columns 

# In[ ]:

preprocess_data_inplace(test_df)


# In[ ]:

print test_df.shape
test_df.tail()


# In[ ]:




# Make predictions

# In[ ]:

X_test = test_df[features]


# In[ ]:

X_test = StandardScaler().fit_transform(X_test)


# In[ ]:

Y_pred = model.predict(X_test)


# In[ ]:

Y_pred[:5]


# In[ ]:



