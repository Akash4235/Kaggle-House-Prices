#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
from scipy.stats import skew


# In[3]:


# Importing data.
train = pd.read_csv(r"C:\Users\aakas\Documents\competetion\house price pred\train.csv")
test = pd.read_csv(r"C:\Users\aakas\Documents\competetion\house price pred\test.csv")
y = train['SalePrice']
train.drop('SalePrice',axis=1, inplace=True)
df = pd.concat([train,test])
df = df.reset_index(drop=True)


# ## Missing Values Treatment

# In[4]:


# Data points whose only 1 or 2 values are missing are filled using mode.
lst = ['BsmtHalfBath','Functional','Utilities','BsmtFullBath','Electrical','BsmtFinSF1','Exterior1st',
       'Exterior2nd','GarageCars','GarageArea','KitchenQual','SaleType','TotalBsmtSF','BsmtUnfSF','BsmtFinSF2','ExterQual']
for i in lst:
    df[i].replace(np.nan, df[i].mode()[0], inplace=True)
# columns with no garge values.
gar_fill = ['GarageType','GarageCond','GarageFinish','GarageQual']

for i in gar_fill:
    df[i].replace(np.nan, 'no garage',inplace=True)
# columns with missing basement values.
    bsmt_fill = ['BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual']

for i in bsmt_fill:
    df[i].replace(np.nan, 'no bsmt',inplace=True)
# Some other columns with none value.
df['Alley'].fillna(value='no alley',inplace=True)
df['PoolQC'].fillna(value='No pool',inplace=True)
df['MiscFeature'].fillna(value='no feature',inplace=True)
df['Fence'].fillna(value='no fence',inplace=True)
df['FireplaceQu'].fillna(value='no fireplace',inplace=True)
# Some more columns to fill.
df['GarageYrBlt'].fillna(value=0,inplace=True)
df['MasVnrArea'].fillna(value=0, inplace=True)
df['MasVnrType'].fillna(value='None',inplace=True)
df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))


# In[5]:


# Filling some categorical values manually.
value_map = {'TA':5,'Gd':7,'Ex':9,'Fa':3,'no bsmt':0,'Po':2,'GLQ':1, 'ALQ':2, 'Unf':3, 'Rec':4, 'BLQ':5, 'LwQ':6,
            'no fireplace':0,'no garage':0,'No pool':0,'no fence':0, 'MnPrv':3, 'GdWo':2, 'GdPrv':4, 'MnWw':1}

df['KitchenQual'] = df['KitchenQual'].map(value_map).astype('int')

df['BsmtCond'] = df['BsmtCond'].map(value_map).astype('int')
df['BsmtQual'] = df['BsmtQual'].map(value_map).astype('int')

df['BsmtFinType1'] = df['BsmtFinType1'].map(value_map).astype('int')
df['BsmtFinType2'] = df['BsmtFinType2'].map(value_map).astype('int')

df['FireplaceQu'] = df['FireplaceQu'].map(value_map).astype('int')

df['GarageCond'] = df['GarageCond'].map(value_map).astype('int')
df['GarageQual'] = df['GarageQual'].map(value_map).astype('int')

df['PoolQC'] = df['PoolQC'].map(value_map).astype('int')

df['Fence'] = df['Fence'].map(value_map).astype('int')

df['ExterCond'] = df['ExterCond'].map(value_map).astype('int')
df['ExterQual'] = df['ExterQual'].map(value_map).astype('int')
df['HeatingQC'] = df['HeatingQC'].map(value_map).astype('int')


# ## Handling Skewed Data

# In[6]:


# Calling numerical dtypes data from the original data to correct skewness.
numerical = [x for x in dict(df.dtypes) if dict(df.dtypes)[x] in ['float64','int64']]
numerical_data = df[numerical]
# picking highly skewed columns from the data.
poss_skew = numerical_data.columns
show_skew = np.abs(df[poss_skew].apply(lambda x: skew(x)).sort_values(ascending = False))
high_skew = show_skew[show_skew>0.3] # higher than 0.3 means high skewness.
show_index = high_skew.index

# Function to correct skewness using log function.
for i in show_index:
    df[i] = np.log1p(df[i])


# ## Dropping Outliers

# In[7]:


# Picking training data to plot it and remove outliers.
train_graph = df.iloc[ :len(y), : ]
train_graph = train_graph.join(y)


# In[7]:


# Plotting numerical columns of the data.
sns.pairplot(data=train_graph, y_vars=['SalePrice'], x_vars=['LotFrontage', 'LotArea', 'OverallQual','OverallCond','MasVnrArea'])
sns.pairplot(data=train_graph,y_vars=['SalePrice'],x_vars=['BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','1stFlrSF'])
sns.pairplot(data=train_graph,y_vars=['SalePrice'],x_vars=['2ndFlrSF','LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath'])
sns.pairplot(data=train_graph,y_vars=['SalePrice'],x_vars=['FullBath','HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',])
sns.pairplot(data=train_graph,y_vars=['SalePrice'],x_vars=['Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF'])
sns.pairplot(data=train_graph,y_vars=['SalePrice'],x_vars=['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea'])
sns.pairplot(data=train_graph,y_vars=['SalePrice'],x_vars=[ 'MiscVal','SalePrice'])


# In[8]:


df = df.join(y)
# Joining the y to the data for removing oultliers.
# Shape of the data is (2919,81) before removing outliers.


# In[9]:


# Removing outliers manually using pairplot of the data.
df = df.drop(df[(df['OverallCond']<0.8)&(df['SalePrice']>300000)].index)
df = df.drop(df[(df['OverallCond']>1)&(df['SalePrice']>700000)].index)
df = df.drop(df[(df['OverallQual']>9)&(df['SalePrice']<300000)].index)
df = df.drop(df[(df['MasVnrArea']<1)&(df['SalePrice']>700000)].index)
df = df.drop(df[(df['BsmtHalfBath']>0.5)&(df['SalePrice']>600000)].index)
df = df.drop(df[(df['PoolArea']>1.5)&(df['SalePrice']>600000)].index)


# In[10]:


df.reset_index(drop=True,inplace=True)
y = df['SalePrice']
y.dropna(inplace=True)
df.drop('SalePrice',axis=1,inplace=True)
# Shape of the data after removing outliers (2914,81).


# ## One-Hot Encoding/ dummy Variables

# In[11]:


# Getting categorical data to create Dummy Variables.
catgorical = [x for x in dict(df.dtypes) if dict(df.dtypes)[x] in ['object']]
catgorical_data = df[catgorical]


# In[12]:


cat_dum = pd.get_dummies(catgorical_data, drop_first=True)
df.drop(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'Heating',
       'CentralAir', 'Electrical', 'Functional', 'GarageType', 'GarageFinish',
       'PavedDrive', 'MiscFeature', 'SaleType', 'SaleCondition'],
       axis=1,inplace=True)

df = pd.concat([df,cat_dum],axis=1)
# Shape of the data (2914,215)


# ## Training model

# In[13]:


# Seperating training and test data.
train_f = df.iloc[ :len(y), : ]
train_f = train_f.join(y)
test_f = df.iloc[len(train_f): , : ]


# In[14]:


X = train_f.drop('SalePrice',axis=1)
y = train_f['SalePrice']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state=101)


# In[15]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[16]:


lr.fit(X_train,y_train)


# In[17]:


prediction = lr.predict(X_test)

from sklearn import metrics
from sklearn.metrics import r2_score
print(metrics.mean_absolute_error(y_test, prediction))
print('\n')
print(r2_score(y_test,prediction))


# In[18]:


test_f.reset_index(drop=True,inplace=True)
prediction_test = lr.predict(test_f)


# In[19]:


submission = pd.DataFrame({'Id':test_f['Id'], 'SalePrice':prediction_test})
submission.to_csv('saleprice submission',index=False)
# Error Rate of test set = 0.21


# In[ ]:




