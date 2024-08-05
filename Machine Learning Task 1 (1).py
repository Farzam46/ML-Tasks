#!/usr/bin/env python
# coding: utf-8

# # Importing libraries 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Importing SARIMA model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


# ## Q1:  Import data and check null values, column info, and descriptive statistics of the data.
# 

# In[19]:


# Load the dataset
data = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\Instagram-Reach.csv")

# Display the first few rows of the dataframe
print(data.head())

# Check for null values
print(data.isnull().sum())

# Get column info
print(data.info())

# Descriptive statistics
print(data.describe())


# ## Q2: You can convert the Date column into datetime datatype to move forward.
# 

# In[20]:


data['Date'] = pd.to_datetime(data['Date'])
column_info = data.info()
print(column_info)


# ## Q3: Analyze the trend of Instagram reach over time using a line chart

# In[21]:


plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Instagram reach'])
plt.title('Instagram Reach Over Time')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.show()


# ## Q4: Analyze Instagram reach for each day using a bar chart.

# In[23]:


plt.figure(figsize=(10, 5))
sns.barplot(x=data['Date'], y=data['Instagram reach'])
plt.title('Instagram Reach for Each Day')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.xticks(rotation=45)
plt.show()




# ## Q5:  Analyze the distribution of Instagram reach using a box plot.
# 

# In[24]:


plt.figure(figsize=(10, 5))
sns.boxplot(y=data['Instagram reach'])
plt.title('Distribution of Instagram Reach')
plt.show()


# ## Q6: Now create a day column and analyze reach based on the days of the week. To create a day column, you can use the python method to extract the day of the week from the Date column.

# In[30]:


# Sample DataFrame for illustration
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=30, freq='D'),
    'Instagram reach': [150, 200, 180, 170, 300, 250, 220, 310, 330, 350, 370, 390, 400, 410, 420, 430, 450, 470, 490, 500, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610]
})

# Convert 'Date' column to datetime if not already in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Create 'Day' column
data['Day'] = data['Date'].dt.day_name()

# Group by 'Day'
reach_by_day = data.groupby('Day')['Instagram reach'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Plotting the results
plt.figure(figsize=(12, 6))
sns.barplot(x=reach_by_day.index, y=reach_by_day.values)
plt.title('Instagram Reach by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Instagram Reach')
plt.xticks(rotation=45)
plt.show()


# ## Q7: Now analyze the reach based on the days of the week. For this, you can group the DataFrame by the Day column and calculate the mean, median, and standard deviation of the Instagram reach column for each day.
# 

# In[28]:


# Sample DataFrame for illustration
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=30, freq='D'),
    'Instagram reach': [150, 200, 180, 170, 300, 250, 220, 310, 330, 350, 370, 390, 400, 410, 420, 430, 450, 470, 490, 500, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610]
})

# Convert 'Date' column to datetime if not already in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Create 'Day' column
data['Day'] = data['Date'].dt.day_name()

# Group by 'Day' and calculate mean, median, and standard deviation
reach_by_day = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
print(reach_by_day)

# Plotting the results
plt.figure(figsize=(12, 6))
reach_by_day.plot(kind='bar')
plt.title('Instagram Reach Statistics by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Instagram Reach')
plt.xticks(rotation=45)
plt.show()

# Additional steps for SARIMA forecasting (if needed)

# Decompose the time series
decompose_result = seasonal_decompose(data.set_index('Date')['Instagram reach'], model='additive', period=7)
decompose_result.plot()
plt.show()

# Plot ACF and PACF
plt.figure(figsize=(10, 5))
plot_acf(data['Instagram reach'], lags=10)
plt.title('Autocorrelation Plot')
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(data['Instagram reach'], lags=10)
plt.title('Partial Autocorrelation Plot')
plt.show()

# Example SARIMA model (replace with actual model fitting if required)
model = SARIMAX(data['Instagram reach'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
results = model.fit()

# Forecast future values
forecast = results.get_forecast(steps=7)
forecast_df = forecast.conf_int()
forecast_df['forecast'] = results.predict(start=forecast_df.index[0], end=forecast_df.index[-1])

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Instagram reach'], label='Observed')
plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast')
plt.fill_between(forecast_df.index, forecast_df.iloc[:, 0], forecast_df.iloc[:, 1], color='grey', alpha=0.2)
plt.title('Instagram Reach Forecast')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.legend()
plt.show()


# ## Q8: Now create a bar chart to visualize the reach for each day of the week.

# In[25]:


data['Day'] = data['Date'].dt.day_name()
print(data.head())

# Group by Day and calculate statistics
reach_by_day = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std'])
print(reach_by_day)


# ## Q9:  Check the Trends and Seasonal patterns of Instagram reach.

# In[27]:


# Decompose the time series
decompose_result = seasonal_decompose(data.set_index('Date')['Instagram reach'], model='additive', period=30)
decompose_result.plot()
plt.show()


# ## Q10: You can use the SARIMA model to forecast the reach of the Instagram account. You need to find p, d, and q values to forecast the reach of Instagram. To find the value of d, you can use the autocorrelation plot, and to find the value of q, you can use a partial autocorrelation plot. The value of d will be 1. You have to visualize an autocorrelation plot to find the value of p, partial autocorrelation plot to find the value of q,
# 

# In[26]:


# Plot ACF and PACF
plt.figure(figsize=(10, 5))
plot_acf(data['Instagram reach'], lags=50)
plt.title('Autocorrelation Plot')
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(data['Instagram reach'], lags=50)
plt.title('Partial Autocorrelation Plot')
plt.show()

# Based on these plots, you can determine the values of p and q
# For this example, let's assume p=1 and q=1


# ## Q11: You have to train a model using SARIMA and make prediction

# In[31]:


# Define the SARIMA model
model = SARIMAX(data['Instagram reach'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Forecast future values
forecast = results.get_forecast(steps=30)
forecast_df = forecast.conf_int()
forecast_df['forecast'] = results.predict(start=forecast_df.index[0], end=forecast_df.index[-1])

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Instagram reach'], label='Observed')
plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast')
plt.fill_between(forecast_df.index, forecast_df.iloc[:, 0], forecast_df.iloc[:, 1], color='grey', alpha=0.2)
plt.title('Instagram Reach Forecast')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.legend()
plt.show()

