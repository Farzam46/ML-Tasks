#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[18]:


df=pd.read_csv("C:/Users/farzam shahzad/Downloads/userbehaviour.csv")


# ## **Question no 1**

# In[8]:


df.head()


# In[16]:


df.isnull()


# In[18]:


df.info()


# In[19]:


df.describe()


# 
# ## **Question no 2**

# In[21]:


df['Average Screen Time'].max()


# In[22]:


df['Average Screen Time'].min()


# In[24]:


df['Average Screen Time'].mean()


# ## **Question no 3**

# In[26]:


df['Average Spent on App (INR)'].mean()


# In[27]:


df['Average Spent on App (INR)'].max()


# In[28]:


df['Average Spent on App (INR)'].min()


# ## **Question no 4**

# In[29]:


import seaborn as sns


# In[19]:


plt.figure(figsize=(10, 5))  
sns.scatterplot(data=df, x='Average Screen Time', y='Average Spent on App (INR)', hue='Last Visited Minutes', alpha=0.6)
plt.title('Relationship Between Spending Capacity and Screen Time')  
plt.xlabel('Average Screen Time')  
plt.ylabel('Average Spent on App')  
plt.grid(True)  
plt.show()


# ## **Question no 5**

# In[43]:


plt.figure(figsize=(10, 6))  
sns.scatterplot(data=df, x='Ratings', y='Average Screen Time')
plt.title('Relationship Between Rating  and Screen Time')  
plt.xlabel('Ratings')  
plt.ylabel('Average Screen time')  
plt.grid(True)  
plt.show()


# ## **Question no 6**

# In[9]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/farzam shahzad/Downloads/userbehaviour.csv")
features = df[['Average Screen Time', 'Average Spent on App (INR)', 'Last Visited Minutes']]


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


optimal_clusters = 2  

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)


# ## Question no 7

# In[23]:


# visualize these segments:
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average Screen Time', y='Average Spent on App (INR)', hue='Average Screen Time', data=df, palette='viridis')
plt.title('User Segmentation')
plt.xlabel('Average Screen Time (minutes)')
plt.ylabel('Average Spent on App (INR)')
plt.legend(title='Cluster')
plt.show()


# ## Question no 8

# ## Summary:
# # Relationship between Ratings and Average Screen Time:
# A scatter plot was created to visualize the relationship. The correlation coefficient was calculated to quantify the relationship. Observation: The relationship can be analyzed based on the plot and correlation value. App User Segmentation:
# 
# Selected relevant features (Average Screen Time, Average Spent on App (INR), Last Visited Minutes) for clustering. Standardized the features for better clustering performance. Used the elbow method to determine the optimal number of clusters. Applied K-means clustering to segment users into different clusters.
# 
# # Visualization of Segments:
# Created a scatter plot to visualize the user segments based on Average Screen Time and Average Spent on App (INR).
# 
# The number of segments obtained from the K-means clustering will help in understanding the different user groups. For example, users with high screen time and high spending could be considered retained, while users with low screen time and low spending could be considered lost.
