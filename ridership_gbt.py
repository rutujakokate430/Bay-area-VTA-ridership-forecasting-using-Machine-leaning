# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    make_scorer,
)
import pandas as pd
import numpy as np
import pickle
import os

# %%
train_df = pd.read_csv("clean_data/train.csv")
test_df = pd.read_csv("clean_data/test.csv")

# %%
X_train = train_df[[x for x in train_df.columns if x not in ["On", "Off"]]]
Y_train = train_df["On"]

# %%
X_test = test_df[[x for x in test_df.columns if x not in ["On", "Off"]]]
Y_test = test_df["On"]

# %%
gbr = GradientBoostingRegressor(random_state = 42, verbose=2)
gbr.fit(X_train, Y_train)

# %%
Y_pred_gbr = np.floor(gbr.predict(X_test)).astype(int)

# %%
rmse_gbr = float(format(np.sqrt(mean_squared_error(Y_test, Y_pred_gbr)), '.3f'))

# %%
rmse_gbr

# %%
MODELS_FOLDER = "models"

# %%
pickle.dump(gbr, open(os.path.join(MODELS_FOLDER, "base_gbt.pkl"), 'wb'))

# %%
r2 = r2_score(Y_test, Y_pred_gbr)
rmse = root_mean_squared_error(Y_test, Y_pred_gbr)
mae = mean_absolute_error(Y_test, Y_pred_gbr)
print(f"test rmse: {rmse}, mae: {mae}, r2: {r2}")

# %%
Y_pred_gbr = np.floor(gbr.predict(X_train)).astype(int)
r2 = r2_score(Y_train, Y_pred_gbr)
rmse = root_mean_squared_error(Y_train, Y_pred_gbr)
mae = mean_absolute_error(Y_train, Y_pred_gbr)
print(f"train rmse: {rmse}, mae: {mae}, r2: {r2}")

# %%
base_gbt = GradientBoostingRegressor(
    random_state=42,
    loss="squared_error",
    criterion="friedman_mse",
    min_samples_split=14,
    min_samples_leaf=7,
    verbose=2,
)
param_grid = [
    {
        "learning_rate": [0.001, 0.1],
        "subsample": [0.8, 1.0],
        "n_estimators": [50],
        "max_depth": [20],
        "max_features": [1.0],
    },
    {
        "learning_rate": [0.001, 0.1],
        "subsample": [0.8, 1.0],
        "n_estimators": [50],
        "max_depth": [100],
        "max_features": ["sqrt"],
    },
    {
        "learning_rate": [0.001, 0.1],
        "subsample": [0.8, 1.0],
        "n_estimators": [10],
        "max_depth": [70],
        "max_features": [1.0],
    },
    {
        "learning_rate": [0.001, 0.1],
        "subsample": [0.8, 1.0],
        "n_estimators": [10],
        "max_depth": [100],
        "max_features": ["sqrt"],
    },
]

# %%
scorer = make_scorer(mean_squared_error, greater_is_better=False)
grid_search = GridSearchCV(base_gbt, param_grid, scoring=scorer, n_jobs=-1, verbose=2, cv=3)

# %%
grid_search.fit(X_train, Y_train)

# %%
pd.DataFrame(grid_search.cv_results_)

# %%
print(grid_search.best_params_)

# %%
best_gbt = grid_search.best_estimator_

# %%
train_y_pred = np.floor(best_gbt.predict(X_train)).astype(int)
test_y_pred = np.floor(best_gbt.predict(X_test)).astype(int)

# %%
print("train rmse:", root_mean_squared_error(Y_train, train_y_pred))
print("train mae:", mean_absolute_error(Y_train, train_y_pred))
print("train r2 score:", r2_score(Y_train, train_y_pred))

# %%
print("test rmse:", root_mean_squared_error(Y_test, test_y_pred))
print("test mae:", mean_absolute_error(Y_test, test_y_pred))
print("test r2 score:", r2_score(Y_test, test_y_pred))

# %%
pickle.dump(best_gbt, open(os.path.join(MODELS_FOLDER, "tuned_gbt.pkl"), 'wb'))


