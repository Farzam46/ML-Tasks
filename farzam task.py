#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load the dataset
file_path = "C:\\Users\\LENOVO\\Desktop\\user_profiles_for_ads.csv"
df = pd.read_csv(file_path)

# Check for null values
null_values = df.isnull().sum()
print("Null Values:\n", null_values)

# Get column information
column_info = df.info()

# View descriptive statistics
descriptive_stats = df.describe(include='all')
print("Descriptive Statistics:\n", descriptive_stats)


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of age
plt.figure(figsize=(10, 6))
sns.countplot(y='Age', data=df, order=df['Age'].value_counts().index)
plt.title('Distribution of Age')
plt.xlabel('Count')
plt.ylabel('Age Range')
plt.show()

# Visualize the distribution of gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=df)
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Visualize the distribution of education level
plt.figure(figsize=(12, 6))
sns.countplot(y='Education Level', data=df, order=df['Education Level'].value_counts().index)
plt.title('Distribution of Education Level')
plt.xlabel('Count')
plt.ylabel('Education Level')
plt.show()

# Visualize the distribution of income level
plt.figure(figsize=(12, 6))
sns.countplot(y='Income Level', data=df, order=df['Income Level'].value_counts().index)
plt.title('Distribution of Income Level')
plt.xlabel('Count')
plt.ylabel('Income Level')
plt.show()


# In[4]:


# Visualize device usage patterns
plt.figure(figsize=(10, 6))
sns.countplot(x='Device Usage', data=df)
plt.title('Device Usage Patterns')
plt.xlabel('Device Type')
plt.ylabel('Count')
plt.show()

# Analyze users' online behavior
# Time spent online on weekdays
plt.figure(figsize=(10, 6))
sns.histplot(df['Time Spent Online (hrs/weekday)'], bins=20, kde=True)
plt.title('Time Spent Online on Weekdays')
plt.xlabel('Hours')
plt.ylabel('Count')
plt.show()

# Time spent online on weekends
plt.figure(figsize=(10, 6))
sns.histplot(df['Time Spent Online (hrs/weekend)'], bins=20, kde=True)
plt.title('Time Spent Online on Weekends')
plt.xlabel('Hours')
plt.ylabel('Count')
plt.show()

# Click-through rates (CTR)
plt.figure(figsize=(10, 6))
sns.histplot(df['Click-Through Rates (CTR)'], bins=20, kde=True)
plt.title('Click-Through Rates (CTR)')
plt.xlabel('CTR (%)')
plt.ylabel('Count')
plt.show()

# Conversion rates
plt.figure(figsize=(10, 6))
sns.histplot(df['Conversion Rates'], bins=20, kde=True)
plt.title('Conversion Rates')
plt.xlabel('Conversion Rate (%)')
plt.ylabel('Count')
plt.show()

# Ad interaction time
plt.figure(figsize=(10, 6))
sns.histplot(df['Ad Interaction Time (sec)'], bins=20, kde=True)
plt.title('Ad Interaction Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Count')
plt.show()

# Identify the most common interests among users
plt.figure(figsize=(12, 6))
interests = df['Top Interests'].str.get_dummies(sep=',').sum().sort_values(ascending=False)
sns.barplot(x=interests.values, y=interests.index)
plt.title('Most Common Interests Among Users')
plt.xlabel('Count')
plt.ylabel('Interests')
plt.show()


# In[5]:


# Analyze the average time spent online on weekdays vs. weekends
avg_time_weekday = df['Time Spent Online (hrs/weekday)'].mean()
avg_time_weekend = df['Time Spent Online (hrs/weekend)'].mean()

print(f"Average Time Spent Online on Weekdays: {avg_time_weekday:.2f} hours")
print(f"Average Time Spent Online on Weekends: {avg_time_weekend:.2f} hours")

# Visualize the average time spent online
time_data = pd.DataFrame({
    'Day Type': ['Weekdays', 'Weekends'],
    'Average Time Spent (hours)': [avg_time_weekday, avg_time_weekend]
})

plt.figure(figsize=(8, 6))
sns.barplot(x='Day Type', y='Average Time Spent (hours)', data=time_data)
plt.title('Average Time Spent Online: Weekdays vs. Weekends')
plt.xlabel('Day Type')
plt.ylabel('Average Time Spent (hours)')
plt.show()

# Investigate user engagement metrics: likes and reactions
plt.figure(figsize=(10, 6))
sns.histplot(df['Likes and Reactions'], bins=20, kde=True)
plt.title('Distribution of Likes and Reactions')
plt.xlabel('Likes and Reactions')
plt.ylabel('Count')
plt.show()

# Analyze ad interaction metrics
# Click-Through Rates (CTR)
plt.figure(figsize=(10, 6))
sns.histplot(df['Click-Through Rates (CTR)'], bins=20, kde=True)
plt.title('Click-Through Rates (CTR)')
plt.xlabel('CTR (%)')
plt.ylabel('Count')
plt.show()

# Conversion Rates
plt.figure(figsize=(10, 6))
sns.histplot(df['Conversion Rates'], bins=20, kde=True)
plt.title('Conversion Rates')
plt.xlabel('Conversion Rate (%)')
plt.ylabel('Count')
plt.show()

# Ad Interaction Time
plt.figure(figsize=(10, 6))
sns.histplot(df['Ad Interaction Time (sec)'], bins=20, kde=True)
plt.title('Ad Interaction Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Count')
plt.show()


# In[6]:


# Identify and visualize the most common interests
# Split the 'Top Interests' column into individual interests
interests_series = df['Top Interests'].str.split(',', expand=True).stack()

# Count the frequency of each interest
interest_counts = interests_series.value_counts()

# Visualize the most common interests
plt.figure(figsize=(12, 8))
sns.barplot(x=interest_counts.values, y=interest_counts.index)
plt.title('Most Common Interests Among Users')
plt.xlabel('Count')
plt.ylabel('Interests')
plt.show()


# In[12]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans



# Define categorical and numeric features
categorical_features = ['Gender', 'Location', 'Education Level']
numeric_features = ['Age', 'Income Level', 'Time Spent Online (hrs/weekday)', 
                    'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 
                    'Click-Through Rates (CTR)', 'Conversion Rates']

# Create a preprocessing pipeline for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Create a preprocessing pipeline for numeric data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply preprocessing
X = preprocessor.fit_transform(df)

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Apply K-Means clustering
optimal_k = 4  # Choose optimal K based on the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X)

# Add the cluster labels to the original dataframe
df['Cluster'] = clusters

# Visualize the distribution of clusters
plt.figure(figsize=(12, 8))
sns.countplot(x='Cluster', data=df)
plt.title('Distribution of User Segments')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

# Analyze the clusters
cluster_analysis = df.groupby('Cluster').mean()
print(cluster_analysis)


# In[13]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# Define categorical and numeric features
categorical_features = ['Gender', 'Location', 'Education Level']
numeric_features = ['Age', 'Income Level', 'Time Spent Online (hrs/weekday)', 
                    'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 
                    'Click-Through Rates (CTR)', 'Conversion Rates']

# Create a preprocessing pipeline for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Create a preprocessing pipeline for numeric data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply preprocessing
X = preprocessor.fit_transform(df)

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Apply K-Means clustering
optimal_k = 4  # Choose optimal K based on the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X)

# Add the cluster labels to the original dataframe
df['Cluster'] = clusters

# Compute mean values for numerical features
numerical_cluster_means = df.groupby('Cluster')[numeric_features].mean()
print("Mean values of numerical features within each cluster:")
print(numerical_cluster_means)

# Compute mode for categorical features
categorical_cluster_modes = df.groupby('Cluster')[categorical_features].agg(lambda x: x.mode().values[0])
print("Mode values of categorical features within each cluster:")
print(categorical_cluster_modes)


# In[14]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans



# Define categorical and numeric features
categorical_features = ['Gender', 'Location', 'Education Level']
numeric_features = ['Age', 'Income Level', 'Time Spent Online (hrs/weekday)', 
                    'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 
                    'Click-Through Rates (CTR)', 'Conversion Rates']

# Create a preprocessing pipeline for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Create a preprocessing pipeline for numeric data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply preprocessing
X = preprocessor.fit_transform(df)

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Apply K-Means clustering
optimal_k = 5  # Choose optimal K based on the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X)

# Add the cluster labels to the original dataframe
df['Cluster'] = clusters

# Compute mean values for numerical features
numerical_cluster_means = df.groupby('Cluster')[numeric_features].mean()

# Compute mode for categorical features
categorical_cluster_modes = df.groupby('Cluster')[categorical_features].agg(lambda x: x.mode().values[0])

# Display cluster means and modes
print("Mean values of numerical features within each cluster:")
print(numerical_cluster_means)
print("\nMode values of categorical features within each cluster:")
print(categorical_cluster_modes)

# Assign cluster names based on characteristics
cluster_names = {
    0: "Weekend Warriors",
    1: "Engaged Professionals",
    2: "Low-Key Users",
    3: "Active Explorers",
    4: "Budget Browsers"
}

# Print cluster characteristics with names
for cluster in range(optimal_k):
    print(f"\nCluster {cluster} – {cluster_names[cluster]}")
    print("Mean values of numerical features:")
    print(numerical_cluster_means.loc[cluster])
    print("Mode values of categorical features:")
    print(categorical_cluster_modes.loc[cluster])

# Add cluster names to the dataframe
df['Cluster_Name'] = df['Cluster'].map(cluster_names)

# Save the updated dataframe to a new CSV file if needed
df.to_csv('path/to/user_profiles_with_clusters.csv', index=False)


# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi



# Define numerical features
numeric_features = ['Age', 'Income Level', 'Time Spent Online (hrs/weekday)', 
                    'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 
                    'Click-Through Rates (CTR)', 'Conversion Rates']

# Compute mean values for numerical features
numerical_cluster_means = df.groupby('Cluster')[numeric_features].mean()

# Normalize data for radar chart
def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

normalized_means = normalize(numerical_cluster_means)

# Number of variables
categories = normalized_means.columns
num_vars = len(categories)

# Radar chart function
def plot_radar(data, labels, title):
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for i, (cluster, values) in enumerate(data.iterrows()):
        values = values.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Cluster {i} - {labels[i]}')
        ax.fill(angles, values, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, rotation=45, ha='right')
    
    plt.title(title, size=15, color='blue', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()

# Cluster names
cluster_names = {
    0: "Weekend Warriors",
    1: "Engaged Professionals",
    2: "Low-Key Users",
    3: "Active Explorers",
    4: "Budget Browsers"
}

# Prepare data for radar chart
data_to_plot = normalized_means
labels = [cluster_names[i] for i in range(len(data_to_plot))]

# Plot radar chart
plot_radar(data_to_plot, labels, 'Cluster Comparison Across Selected Features')


# ##### In the user profiling and segmentation analysis, we imported and cleaned the dataset, visualized key demographic and behavioral variables, and applied K-Means clustering to segment users into distinct groups based on features such as online activity, engagement metrics, and interests. We identified five clusters: "Weekend Warriors," "Engaged Professionals," "Low-Key Users," "Active Explorers," and "Budget Browsers," each characterized by specific demographic and behavioral traits. To visualize these segments, we created a radar chart comparing the mean values of selected features across clusters, providing a clear representation of each segment's profile. This process enabled us to understand user behavior and preferences, facilitating more effective and targeted ad campaigns.
