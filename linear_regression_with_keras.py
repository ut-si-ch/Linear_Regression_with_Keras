#!/usr/bin/env python
# coding: utf-8

# In[3]:





# # Linear Regression Using keras

# In[3]:





# # **1. Objective:**
# In this notebook, a linear regression model is used to predict the fuel efficiency of the late-1970s and early 1980s automobiles.In a regression problem, the aim is to predict the output of a continuous value like a price or a height or a weight,  etc

# In[3]:





# # DataSet Reference:- https://www.kaggle.com/datasets/uciml/autompg-dataset

# In[3]:





# # Importing Libraries

# In[4]:


import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[5]:


# Check the version of Tensorflow and Keras
print("Tensorflow_version:",tf.__version__)
print("Keras_Version:",tf.keras.__version__)


# # Load the Dataset

# In[6]:


# Load the dataset
mpg_df = pd.read_csv("/content/sample_data/auto-mpg.csv")


# In[7]:


# Print the first 5 rows of dataset
mpg_df.head()


# In[8]:


# Print the last 5 rows of dataset
mpg_df.tail()


# In[9]:


# check the shape of the dataset
mpg_df.shape


# In[10]:


# Check the info of the dataset
mpg_df.info()


# In[11]:


# Check the number of categories in the cyclinder dataset
mpg_df["cylinders"].value_counts()


# In[12]:


# Check the number of categories in 'Model Year' feature
mpg_df["model year"].value_counts()


# In[13]:


# Check the number of categories in "Origin feature"
mpg_df["origin"].unique()


# In[14]:


# check the unique category in Horse power column
mpg_df["horsepower"].value_counts()


# # Clean and inspect the data

# In[15]:


# Check the null values in the dataset
mpg_df.isna().sum()


# In[16]:


# Check the overall statistics of the dataset
mpg_df.describe().transpose()


# In[17]:


# Convert 'horsepower' to numeric, coercing errors to NaN
mpg_df['horsepower'] = pd.to_numeric(mpg_df['horsepower'], errors='coerce')

# Drop any rows with NaN values
mpg_df.dropna(inplace=True)


# In[18]:


mpg_df[mpg_df["mpg"] > 46]


# In[19]:


# function to plot the comparison between "MPG" and other numerical features
def plot(feature):
  plt.figure(figsize=(15,8))
  plt.scatter(mpg_df[feature],mpg_df["mpg"], label='Data')
  plt.xlabel(feature)
  plt.ylabel("MPG")
  plt.legend()
  plt.show()


# In[20]:


# Compare the MPG vs HorsePower
plot("horsepower")


# In[21]:


# Compare the "MPG" VS Weight
plot("weight")


# In[21]:





# # Prepare the data

# In[22]:


# Separate the input variable and target variable
input_features = mpg_df.drop(["mpg",'car name'], axis=1)
target = mpg_df["mpg"]


# In[23]:


# Import libraries for splitting the dataset into train and test split
from sklearn.model_selection import train_test_split


# In[24]:


# Split the dataset into train set = 80 % and test set 20%.
X_train, X_test, y_train, y_test = train_test_split(input_features, target, test_size=0.2, random_state=42)


# In[25]:


# Check the shape of X_train and X_test
input_features.shape, X_train.shape, X_test.shape


# In[26]:


X_train.head()


# In[27]:


# Normalize the numerical features using keras normalization
from tensorflow.keras.layers import Normalization


# In[28]:


# The first step is to create a normalization layer
normalizer = Normalization()


# In[29]:


# Then fit the state of the preprocessing layer to the data by calling Normalization.adpat .
normalizer.adapt(np.array(X_train))


# In[30]:


# Calculate the mean and variance, and store them in the layer
print("Mean:",normalizer.mean.numpy())
print("Variance:",normalizer.variance.numpy())


# In[31]:


# When the layer is called, it returns the input data, with each feature independently normalized.
first = np.array(X_train[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


# In[31]:





# # Building a Linear Regression Neural Network

# In[32]:


# import libraries for building sequential models with dense layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


# In[33]:


# define the sequential model
model = Sequential()

# add the input layer with 7 features
model.add(tf.keras.Input(shape= (7,), name = 'Input Layer'))
model.add(normalizer)

# Add the 2 dense layer
model.add(Dense(64, activation="relu", name = 'Hidden_Layer1'))
model.add(Dense(64, activation="relu", name = 'Hidden_Layer2'))

# Add the outpt layer
model.add(Dense(1, name = 'Output_Layer'))


# In[34]:


# Print the summary of a model
model.summary()


# In[35]:


# plotting the model
from tensorflow.keras.utils import plot_model
plot_model(model)


# In[35]:





# # Compile the Neural Network

# In[36]:


# Compile the model with 0.001 as learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mae")


# In[36]:





# # Training a Neural Network/

# In[37]:


# define the batch size and number of epochs
batch_size = 16
num_epochs = 100

# fit a neural network or train a neural network model
linear_regressor = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1,validation_split=0.2)


# In[38]:


# prompt: why i am getting loss = nan and val_loss as nan values

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0) # Clip gradients to a max norm of 1.0
model.compile(optimizer=optimizer, loss="mae")


# In[38]:





# # Evaluate a trained network

# In[39]:


# Evaluate our neural network using test dataset
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score)


# In[40]:


# Plot training loss and validation loss
plt.plot(linear_regressor.history['loss'], label='Training Loss')
plt.plot(linear_regressor.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()


# In[40]:





# # Inference on New data

# In[41]:


# prediction on test dataset using predict() method
y_pred = model.predict(X_test)


# In[42]:


# Check predictions on first 10 datapoints from X_test
y_pred[:10]


# In[43]:


# Convert or flattern 2d array to 1d array
y_pred = y_pred.flatten()


# In[44]:


# Compare the actual and prediction values for first 10 datapoints
print("Actual Values:",np.array(y_test[:10]))
print("Predicted Values:",y_pred[:10])


# In[45]:


# check how best fit the line is?
a = plt.axes(aspect = 'equal')
plt.scatter(y_test, y_pred)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0,50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)


# In[46]:


# Check the distribution of errors
error = y_pred - y_test
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")


# In[46]:




