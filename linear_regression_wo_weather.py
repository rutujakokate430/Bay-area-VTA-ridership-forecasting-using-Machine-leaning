# %% [markdown]
# # Linear Regression

# %% [markdown]
# ## Import packages

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer,
)
import pandas as pd
import numpy as np
import pickle
import os

# %%
CLEAN_DATA_FOLDER = "clean_data"
MODELS_FOLDER = "models"

# %% [markdown]
# ## Load the dataframe

# %%
train_df = pd.read_csv(os.path.join(CLEAN_DATA_FOLDER, "train_wo_weather.csv"))
test_df = pd.read_csv(os.path.join(CLEAN_DATA_FOLDER, "test_wo_weather.csv"))

# %% [markdown]
# ## Split into X and y

# %%
train_X = train_df[[x for x in train_df.columns if x not in ["On", "Off"]]]
train_y = train_df["On"]
test_X = test_df[[x for x in test_df.columns if x not in ["On", "Off"]]]
test_y = test_df["On"]

# %% [markdown]
# ## Train the Linear Regression Model

# %%
std = StandardScaler()
pca = PCA(random_state=42)
reg = ElasticNet(random_state=42)
lr = Pipeline(
    [("standardization", std), ("decomposition", pca), ("regression", reg)],
    verbose=True,
)
lr = lr.fit(X=train_X, y=train_y)

# %%
train_y_pred = np.floor(lr.predict(train_X)).astype(int)
test_y_pred = np.floor(lr.predict(test_X)).astype(int)

# %% [markdown]
# ## Report Train and Test results

# %%
print("train rmse:", root_mean_squared_error(train_y, train_y_pred))
print("train mae:", mean_absolute_error(train_y, train_y_pred))
print("train r2 score:", r2_score(train_y, train_y_pred))

# %%
print("test rmse:", root_mean_squared_error(test_y, test_y_pred))
print("test mae:", mean_absolute_error(test_y, test_y_pred))
print("test r2 score:", r2_score(test_y, test_y_pred))

# %% [markdown]
# ## Export Model

# %%
pickle.dump(lr, open(os.path.join(MODELS_FOLDER, "base_elastic_net_wo_weather.pkl"), "wb"))

# %% [markdown]
# ## Hyperparameter Tuning with GridSearchCV

# %% [markdown]
# ### Declare base model and parameters

# %%
std = StandardScaler()
pca = PCA(random_state=42)
reg = ElasticNet(random_state=42)
base_lr = Pipeline(
    [("standardization", std), ("decomposition", pca), ("regression", reg)],
    verbose=True,
)
param_grid = {
    "decomposition__n_components": [5, 6, 7],
    "regression__alpha": [0.5, 1.0],
    "regression__l1_ratio": [0.3, 0.5, 0.7],
}

# %% [markdown]
# ### Declare the scorer and grid search

# %%
scorer = make_scorer(mean_squared_error, greater_is_better=False)
grid_search = GridSearchCV(base_lr, param_grid, scoring=scorer, n_jobs=-1, verbose=2, cv=5)

# %% [markdown]
# ### Train the models

# %%
grid_search.fit(train_X, train_y)

# %%
pd.DataFrame(grid_search.cv_results_)

# %%
print(grid_search.best_params_)

# %% [markdown]
# ### Extract the best model

# %%
best_lr = grid_search.best_estimator_

# %%
train_y_pred = np.floor(best_lr.predict(train_X)).astype(int)
test_y_pred = np.floor(best_lr.predict(test_X)).astype(int)

# %% [markdown]
# ### Report Train and Test results

# %%
print("train rmse:", root_mean_squared_error(train_y, train_y_pred))
print("train mae:", mean_absolute_error(train_y, train_y_pred))
print("train r2 score:", r2_score(train_y, train_y_pred))

# %%
print("test rmse:", root_mean_squared_error(test_y, test_y_pred))
print("test mae:", mean_absolute_error(test_y, test_y_pred))
print("test r2 score:", r2_score(test_y, test_y_pred))

# %% [markdown]
# ### Export Models

# %%
pickle.dump(best_lr, open(os.path.join(MODELS_FOLDER, "tuned_elastic_net_wo_weather.pkl"), "wb"))


