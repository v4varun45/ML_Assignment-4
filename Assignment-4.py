#!/usr/bin/env python
# coding: utf-8

# # Linear Regression,

# In[33]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# In[13]:


Salary=pd.read_csv('Salary_Data.csv')


# In[4]:


Salary.head()


# In[5]:


Salary.tail()


# In[15]:


X = Salary.iloc[:, :-1].values
y = Salary.iloc[:, 1].values


# In[16]:


y


# In[17]:


X


# In[18]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


# In[51]:


from sklearn.linear_model import LinearRegression


# In[52]:


lin_Reg = LinearRegression()


# In[53]:


lin_Reg.fit(X_train.reshape(-1, 1),y_train)
y_pred = lin_Reg.predict(X_test)
y_pred


# In[54]:


print('Mean Square Error:',metrics.mean_squared_error(y_test,y_pred))


# In[55]:


plt.title('Training data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(X_train, y_train)
plt.show()


# In[56]:


plt.title('Testing data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(X_test, y_test)
plt.show()


# # K means clustering

# In[26]:


KMean=pd.read_csv("K-Mean_Dataset.csv")


# In[27]:


KMean.head()


# In[28]:


KMean.tail()


# In[31]:


X = KMean.iloc[:,1:].values
Mean = SimpleImputer(missing_values=np.nan, strategy='mean')
Mean = Mean.fit(X)
X = Mean.transform(X)


# In[ ]:





# In[43]:



val = []
for i in range(1,11):
    kmeans_cluster = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans_cluster.fit(X)
    val.append(kmeans_cluster.inertia_)

plt.plot(range(1,11),val)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('val')
plt.show()


# In[44]:


Nclusters = 4
Mdl = KMeans(n_clusters=Nclusters)
Mdl.fit(X)


# In[45]:


y_cluster = Mdl.predict(X)
Score = metrics.silhouette_score(X, y_cluster)
print('Silhouette score:',Score)


# # Feature Scaling

# In[48]:


Scaler = preprocessing.StandardScaler()
Scaler.fit(X)
X_scaled_array = Scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled_array)
nclusters = 4 
Mdl = KMeans(n_clusters=nclusters)
Mdl.fit(X_scaled)


# In[50]:


y_scaled_cluster_kmeans = Mdl.predict(X_scaled)
score_ = metrics.silhouette_score(X_scaled, y_scaled_cluster_kmeans)
print('Silhouette score: ',score_)


# In[ ]:


Silhouette score decreased after doing feature scaling

