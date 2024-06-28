# %% [markdown]
# # Random Forest Regression

# %% [markdown]
# ## Import packages

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error,
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
train_df = pd.read_csv(os.path.join(CLEAN_DATA_FOLDER, "train.csv"))
test_df = pd.read_csv(os.path.join(CLEAN_DATA_FOLDER, "test.csv")).sort_values(
    ["Day", "Line", "Service", "Direction Number", "Sequence"]
)

# %% [markdown]
# ## Split into X and y

# %%
train_X = train_df[[x for x in train_df.columns if x not in ["On", "Off"]]]
train_y = train_df["On"]
test_X = test_df[[x for x in test_df.columns if x not in ["On", "Off"]]]
test_y = test_df["On"]

# %% [markdown]
# ## Train the Random Forest Regressor Model

# %%
rf = RandomForestRegressor(
    n_estimators=70,
    max_depth=10,
    random_state=42,
    min_samples_split=14,
    min_samples_leaf=7,
    n_jobs=-1,
    verbose=1,
    criterion="poisson",
)
rf = rf.fit(X=train_X, y=train_y)

# %%
train_y_pred = np.floor(rf.predict(train_X)).astype(int)
test_y_pred = np.floor(rf.predict(test_X)).astype(int)

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
pickle.dump(rf, open(os.path.join(MODELS_FOLDER, "base_random_forest.pkl"), "wb"))

# %%
del rf

# %% [markdown]
# ## Hyperparameter Tuning with GridSearchCV

# %% [markdown]
# | n_estimators | max_depth | max_features | criterion |
# | --- | --- | --- | --- |
# | 100 | 10, 20 | None | squared_error, poisson |
# | 100 | 70, None | sqrt | squared_error, poisson |
# | 10 | 70 | 1.0 | squared_error, poisson |
# | 10 | None | sqrt | squared_error, poisson |

# %% [markdown]
# ### Declare base model and parameters

# %%
base_rf = RandomForestRegressor(random_state=42, min_samples_split=14, min_samples_leaf=7, n_jobs=-1)
param_grid = [
    {
        "criterion": ["squared_error", "poisson"],
        "n_estimators": [50],
        "max_depth": [20],
        "max_features": [1.0],
    },
    {
        "criterion": ["squared_error", "poisson"],
        "n_estimators": [50],
        "max_depth": [100],
        "max_features": ["sqrt"],
    },
    {
        "criterion": ["squared_error", "poisson"],
        "n_estimators": [10],
        "max_depth": [70],
        "max_features": [1.0],
    },
    {
        "criterion": ["squared_error", "poisson"],
        "n_estimators": [10],
        "max_depth": [100],
        "max_features": ["sqrt"],
    },
]

# %% [markdown]
# ### Declare the scorer and grid search

# %%
scorer = make_scorer(mean_squared_error, greater_is_better=False)
grid_search = GridSearchCV(base_rf, param_grid, scoring=scorer, n_jobs=-1, verbose=2, cv=3)

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
best_rf = grid_search.best_estimator_

# %%
train_y_pred = np.floor(best_rf.predict(train_X)).astype(int)
test_y_pred = np.floor(best_rf.predict(test_X)).astype(int)

# %% [markdown]
# ### Plot the Tree

# %%
# _, ax = plt.subplots(3, 1, figsize=(32, 48))
# for i in range(3):
#     _ = plot_tree(
#         best_rf.estimators_[i],
#         max_depth=4,
#         feature_names=train_X.columns,
#         filled=True,
#         proportion=True,
#         rounded=True,
#         precision=2,
#         fontsize=9,
#         ax=ax[i],
#     )

# %% [markdown]
# ### Feature Importance

# %%
# feat_imp = pd.DataFrame(
#     {
#         "Feature": [x for x in best_rf.feature_names_in_],
#         "Importance": [x for x in best_rf.feature_importances_],
#     }
# )
# _, ax = plt.subplots(1, 1, figsize=(16, 9))
# _ = sns.barplot(feat_imp, x="Feature", y="Importance")
# _ = plt.title("Feature Importance for Best Decision Tree Regressor")

# %% [markdown]
# ### Visualize the Predictions

# %%
# line_fit = pd.DataFrame({"True": test_y, "Predicted": test_y_pred}, index=test_df["Day"])
# _, ax = plt.subplots(1, 1, figsize=(16,9))
# _ = sns.lineplot(line_fit, legend=True, ax=ax)

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
# ### Export Model

# %%
pickle.dump(best_rf, open(os.path.join(MODELS_FOLDER, "tuned_random_forest.pkl"), "wb"))


