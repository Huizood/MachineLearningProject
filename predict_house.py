#!/usr/bin/env python
# coding: utf-8

# # Project Source
# https://www.kaggle.com/datasets/rohit265/housing-sales-factors-influencing-sale-prices

# - Lot_Frontage     - Linear feet of street connected to the property
# - Lot_Area         - Lot size in square feet
# - Bldg_Type        - Type of building
# - House_Style      - Style of the house
# - Overall_Cond     - Overall condition rating of the house
# - Year_Built       - Year the house was built
# - Exter_Cond       - Exterior condition rating of the house
# - Total_Bsmt_SF    - Total square feet of basement area
# - First_Flr_SF     - First-floor square feet
# - Second_Flr_SF    - Second-floor square feet
# - Full_Bath        - Number of full bathrooms
# - Half_Bath        - Number of half bathrooms
# - Bedroom_AbvGr    - Number of bedrooms above ground
# - Kitchen_AbvGr    - Number of kitchens above ground
# - Fireplaces	   - Number of fireplaces
# - Longitude	       - Longitude coordinates of the property location
# - Latitude	       - Latitude coordinates of the property location
# - Sale_Price	   - Sale price of the property
# ## The Label for this project is : Sale_Price

# ### Data Preparation (Cleaning), Exploration & Feature Engineering

# In[105]:


# This is the main cell importing the needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score


# In[106]:


# Reading the data and print out the columns
df = pd.read_csv('housing.csv')
df.head()


# In[107]:


#Checking any missing values
print("\nTotal Missing Values: ", df.isnull().sum().sum())


# In[108]:


# Check Missing Values again
print("Missing Values")
print("Lot_Frontage: ", df['Lot_Frontage'].isnull().sum())
print("Lot_Area: ", df['Lot_Area'].isnull().sum())
print("Bldg_Type: ", df['Bldg_Type'].isnull().sum())
print("House_Style: ", df['House_Style'].isnull().sum())
print("Overall_Cond: ", df['Overall_Cond'].isnull().sum())
print("Year_Built: ", df['Year_Built'].isnull().sum())
print("Exter_Cond: ", df['Exter_Cond'].isnull().sum())
print("Total_Bsmt_SF: ", df['Total_Bsmt_SF'].isnull().sum())
print("First_Flr_SF: ", df['First_Flr_SF'].isnull().sum())
print("Second_Flr_SF: ", df['Second_Flr_SF'].isnull().sum())
print("Full_Bath: ", df['Full_Bath'].isnull().sum())
print("Half_Bath: ", df['Half_Bath'].isnull().sum())
print("Bedroom_AbvGr: ", df['Bedroom_AbvGr'].isnull().sum())
print("Kitchen_AbvGr: ", df['Kitchen_AbvGr'].isnull().sum())
print("Fireplaces: ", df['Fireplaces'].isnull().sum())
print("Longitude: ", df['Longitude'].isnull().sum())
print("Latitude: ", df['Latitude'].isnull().sum())
print("Sale_Price: ", df['Sale_Price'].isnull().sum())


# In[109]:


# Data type stats
df.describe()


# In[110]:


house_style_graph  = df.groupby('House_Style').size()
house_style_graph.plot.pie(autopct='%1.1f%%', startangle=90)

plt.title('House_Style')
plt.ylabel('')
plt.show()


# #### Showing Exter_Cond Graph
# - We found out that the poor is 0.1% which makes poor exter_cond become the outlier.

# In[111]:


exter_graph = df.groupby('Exter_Cond').size()
exter_graph.plot.pie(autopct='%1.1f%%', startangle=90)

plt.title('Exter_Cond Graph')
plt.ylabel('')
plt.show()


# In[112]:


bldg_type_graph  = df.groupby('Bldg_Type').size()
bldg_type_graph.plot.pie(autopct='%1.1f%%', startangle=90)

plt.title('blg_type Graph')
plt.ylabel('')
plt.show()


# In[113]:


year_build = df['Year_Built'].value_counts()
print(year_build)


# In[114]:


year_built_graph = df['Year_Built']
year_built_graph.plot(kind='hist', title = 'Year of Built Buildings')


# In[115]:


sale_price = df['Sale_Price'].value_counts()
print(sale_price)


# ## Graph of Second Floor Square feet
# - Since the majority of the frequncy are giving 0 ft of second floor. Therefore, the people who actually have second floor will be the outliers. Then we should not include this feature in our ML model

# In[116]:


# Graph of Second-floor square feet
Sf_graph = df['Second_Flr_SF']
Sf_graph.plot(kind='hist', title = 'Second Floor Square Feet')


# ### Dropping columns
# - We are dropping 'Fireplaces' since we do not need to use it in our ML model and we drop 'Second_Flr_SF' for eliminate the outlier

# In[117]:


# Drop the unnecessary column, such as fireplace
# Will not use fireplace in our ML model
df = df.drop(columns=['Fireplaces', 'Second_Flr_SF'])
df.head()


# ### Picture the median value of the Sale_Price
# - From the graph, we can see that the median value is about  $150000.

# In[118]:


sns.histplot(df['Sale_Price'], bins=30, kde=True, color='g', edgecolor='black', linewidth=2)

# Add title and labels
plt.title('Target Variable Distribution - Median Value of Homes')
plt.xlabel('Sale Price')
plt.ylabel('Frequency/Density')

# Display the plot
plt.show()


# ## ONE HOT ENCODE
# - We need to turn the non-binary cols in to one hot encode.

# In[122]:


# Specify categorical columns
categorical_columns = ['Bldg_Type', 'House_Style', 'Overall_Cond', 'Exter_Cond']

# Define ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('encode', OneHotEncoder(drop='first'), categorical_columns)
    ],
    remainder='passthrough'  # keep the rest of the columns as is
)

# Prepare the data
X = df.drop('Sale_Price', axis=1)
y = df['Sale_Price']

# Fit and transform the data
X_encoded = preprocessor.fit_transform(X)

# Get feature names after encoding
encoded_feature_names = preprocessor.named_transformers_['encode'].get_feature_names_out(categorical_columns)

# Combine encoded feature names with the rest of the features
# Remainder columns are in their original order after the encoded columns
remainder_columns = list(X.columns.difference(categorical_columns))
all_feature_names = list(encoded_feature_names) + remainder_columns

# Create a DataFrame with the encoded columns and original numerical features
encoded_df = pd.DataFrame(X_encoded, columns=all_feature_names)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(encoded_df, y, test_size=0.2, random_state=42)

# Train the Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.title('Actual vs. Predicted Sale Prices')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.grid(True)
plt.show()


# In[125]:


X = df.drop('Sale_Price', axis=1)
y = df['Sale_Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_df, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Make predictions on the testing set
y_pred_rf = rf_regressor.predict(X_test)

# Plotting the actual vs. predicted sale prices using Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, color='green', alpha=0.5)
plt.title('Actual vs. Predicted Sale Prices (Random Forest)')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.grid(True)
plt.show()


# In[129]:


# Load dataset and preprocess data as needed
# Assuming you have a DataFrame named 'df' with features and target variable

# Split data into features (X) and the target variable (y)
X = df.drop('Sale_Price', axis=1)
y = df['Sale_Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_df, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_regressor, encoded_df, y, cv=5, scoring='neg_mean_squared_error')

# Convert the negative MSE scores to positive and compute RMSE
cv_rmse_scores = np.sqrt(-cv_scores)

# Print the cross-validation RMSE scores
print("Cross-Validation RMSE Scores:", cv_rmse_scores)

# Calculate the mean and standard deviation of the cross-validation scores
print("Mean RMSE:", np.mean(cv_rmse_scores))
print("Standard Deviation of RMSE:", np.std(cv_rmse_scores))

# Fit the Random Forest model on the training data
rf_regressor.fit(X_train, y_train)

# Predict on the testing set
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
      
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.title('Actual vs. Predicted Sale Prices')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.grid(True)
plt.show()


# In[ ]:




