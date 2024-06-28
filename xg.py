# %% [markdown]
# # With Weather Data - Using Base XgBoost Model

# %%
import pickle

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define file paths
folder = "clean_data/"
file_path1 = "train.csv"
file_path2 = "test.csv"

# Load the CSV files into Pandas DataFrames
train_df = pd.read_csv(folder + file_path1)
test_df = pd.read_csv(folder + file_path2)

# Split the train and test datasets into features and target variable
X_train = train_df.drop(columns=["On", "Off"])
y_train = train_df["On"]
X_test = test_df.drop(columns=["On", "Off"])
y_test = test_df["On"]

# Initialize XGBoost regressor
xgb_reg = xgb.XGBRegressor(random_state=42, verbosity=2)

# Train the model
xgb_reg.fit(X_train, y_train)

# %%

# Make predictions
y_pred = np.floor(xgb_reg.predict(X_train)).astype(int)
r2 = r2_score(y_train, y_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
mae = mean_absolute_error(y_train, y_pred)
print(f"train rmse: {rmse}, mae: {mae}, r2: {r2}")

y_pred = np.floor(xgb_reg.predict(X_test)).astype(int)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"test rmse: {rmse}, mae: {mae}, r2: {r2}")

pickle.dump(xgb_reg, open("models/base_xgboost_with_weather.pkl", "wb"))

# %% [markdown]
# # With Weather Data - Using Fine Tuning on XgBoost Model

# %%
import numpy as np
import polars as pl
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, \
    mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, \
    mean_absolute_percentage_error, mean_squared_error

# from google.colab import drive

# Mount Google Drive
# drive.mount('/content/drive')

folder = "clean_data/"
file_path1 = "train.csv"
file_path2 = "test.csv"

# Load the CSV file into a Polars DataFrame
train_df = pd.read_csv(folder + file_path1)
test_df = pd.read_csv(folder + file_path2)

# Convert Polars LazyFrame to pandas DataFrame
# train_df = train_df.collect()
# test_df = test_df.collect().to_pandas()

# Split the Train dataset into features and target variable
X_train = train_df.drop(columns=["On", "Off"])
y_train = train_df["On"]

# Split the test dataset into features and target variable
X_test = test_df.drop(columns=["On","Off"])
y_test = test_df["On"]


# %% [markdown]
# ### Using Grid Search to find the Best Hyperparameters

# %%
# Initialize XGBoost regressor
xgb_reg = xgb.XGBRegressor(random_state=42, verbosity=1)

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.5, 0.7, 1]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                           cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Perform grid search
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# %% [markdown]
# ### Using the Best Model after tuning the hyperparameters for Prediction

# %%
# Get the best model
best_model = grid_search.best_estimator_

# Predict on the test set
# Make predictions
y_pred = np.floor(best_model.predict(X_train)).astype(int)
r2 = r2_score(y_train, y_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
mae = mean_absolute_error(y_train, y_pred)
print(f"train rmse: {rmse}, mae: {mae}, r2: {r2}")

y_pred = np.floor(best_model.predict(X_test)).astype(int)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"test rmse: {rmse}, mae: {mae}, r2: {r2}")

pickle.dump(best_model, open("models/tuned_xgboost_with_weather.pkl", "wb"))

# %%


# %% [markdown]
# # Without Weather data - Using Base Xgboost Model

# %%

# Define file paths
folder = "clean_data/"
file_path1 = "train_wo_weather.csv"
file_path2 = "test_wo_weather.csv"

# Load the CSV files into Pandas DataFrames
train_df = pd.read_csv(folder + file_path1)
test_df = pd.read_csv(folder + file_path2)

# Split the train and test datasets into features and target variable
X_train = train_df.drop(columns=["On", "Off"])
y_train = train_df["On"]
X_test = test_df.drop(columns=["On", "Off"])
y_test = test_df["On"]

# Initialize XGBoost regressor
xgb_reg = xgb.XGBRegressor(random_state=42, verbosity=2)

# Train the model
xgb_reg.fit(X_train, y_train)


# Make predictions
y_pred = np.floor(xgb_reg.predict(X_train)).astype(int)
r2 = r2_score(y_train, y_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
mae = mean_absolute_error(y_train, y_pred)
print(f"train rmse: {rmse}, mae: {mae}, r2: {r2}")

y_pred = np.floor(xgb_reg.predict(X_test)).astype(int)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"test rmse: {rmse}, mae: {mae}, r2: {r2}")

pickle.dump(xgb_reg, open("models/base_xgboost_wo_weather.pkl", "wb"))

# %% [markdown]
# # Without Weather data - Using Fine Tuned Xgboost Model

# %%
# Define file paths
folder = "clean_data/"
file_path1 = "train_wo_weather.csv"
file_path2 = "test_wo_weather.csv"

# Load the CSV files into Pandas DataFrames
train_df = pd.read_csv(folder + file_path1)
test_df = pd.read_csv(folder + file_path2)

# Split the train and test datasets into features and target variable
X_train = train_df.drop(columns=["On", "Off"])
y_train = train_df["On"]
X_test = test_df.drop(columns=["On", "Off"])
y_test = test_df["On"]

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.5, 0.7, 1]
}

# Initialize XGBoost regressor
xgb_reg = xgb.XGBRegressor(random_state=42, verbosity=1)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                           cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)


# Perform grid search
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on the test set
# Make predictions
y_pred = np.floor(best_model.predict(X_train)).astype(int)
r2 = r2_score(y_train, y_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
mae = mean_absolute_error(y_train, y_pred)
print(f"train rmse: {rmse}, mae: {mae}, r2: {r2}")

y_pred = np.floor(best_model.predict(X_test)).astype(int)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"test rmse: {rmse}, mae: {mae}, r2: {r2}")

pickle.dump(best_model, open("models/tuned_xgboost_wo_weather.pkl", "wb"))


