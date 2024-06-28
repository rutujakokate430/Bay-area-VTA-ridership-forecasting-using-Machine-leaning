# %% [markdown]
# # Results

# %% [markdown]
# ## Import packages

# %%
from sklearn.tree import plot_tree
from sklearn.metrics import (
    PredictionErrorDisplay,
    root_mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

# %%
CLEAN_DATA_FOLDER = "clean_data"
MODELS_FOLDER = "models"

# %%
results = {
    ("Train", "RMSE"): {},
    ("Train", "MAE"): {},
    ("Train", "EVS"): {},
    ("Train", "R2"): {},
    ("Test", "RMSE"): {},
    ("Test", "MAE"): {},
    ("Test", "EVS"): {},
    ("Test", "R2"): {},
}

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
# ## ElasticNet

# %%
model = pickle.load(open(os.path.join(MODELS_FOLDER, "base_elastic_net.pkl"), "rb"))

# %%
train_y_pred = np.floor(model.predict(train_X)).astype(int)
test_y_pred = np.floor(model.predict(test_X)).astype(int)

# %%
model_name = "ElasticNet"
results[("Train", "RMSE")][model_name] = root_mean_squared_error(train_y, train_y_pred)
results[("Train", "MAE")][model_name] = mean_absolute_error(train_y, train_y_pred)
results[("Train", "EVS")][model_name] = explained_variance_score(train_y, train_y_pred)
results[("Train", "R2")][model_name] = r2_score(train_y, train_y_pred)
results[("Test", "RMSE")][model_name] = root_mean_squared_error(test_y, test_y_pred)
results[("Test", "MAE")][model_name] = mean_absolute_error(test_y, test_y_pred)
results[("Test", "EVS")][model_name] = explained_variance_score(test_y, test_y_pred)
results[("Test", "R2")][model_name] = r2_score(test_y, test_y_pred)

# %%
_, ax = plt.subplots(1, 2, figsize=(20, 9))
display = PredictionErrorDisplay(y_true=test_y, y_pred=test_y_pred)
_ = display.plot(
    ax[0], kind="actual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)
_ = display.plot(
    ax[1], kind="residual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)

# %%
del model

# %% [markdown]
# ## ElasticNet (tuned)

# %%
model = pickle.load(open(os.path.join(MODELS_FOLDER, "tuned_elastic_net.pkl"), "rb"))

# %%
train_y_pred = np.floor(model.predict(train_X)).astype(int)
test_y_pred = np.floor(model.predict(test_X)).astype(int)

# %%
model_name = "ElasticNet (tuned)"
results[("Train", "RMSE")][model_name] = root_mean_squared_error(train_y, train_y_pred)
results[("Train", "MAE")][model_name] = mean_absolute_error(train_y, train_y_pred)
results[("Train", "EVS")][model_name] = explained_variance_score(train_y, train_y_pred)
results[("Train", "R2")][model_name] = r2_score(train_y, train_y_pred)
results[("Test", "RMSE")][model_name] = root_mean_squared_error(test_y, test_y_pred)
results[("Test", "MAE")][model_name] = mean_absolute_error(test_y, test_y_pred)
results[("Test", "EVS")][model_name] = explained_variance_score(test_y, test_y_pred)
results[("Test", "R2")][model_name] = r2_score(test_y, test_y_pred)

# %%
_, ax = plt.subplots(1, 2, figsize=(20, 9))
display = PredictionErrorDisplay(y_true=test_y, y_pred=test_y_pred)
_ = display.plot(
    ax[0], kind="actual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)
_ = display.plot(
    ax[1], kind="residual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)

# %%
del model

# %% [markdown]
# ## Decision Tree

# %%
model = pickle.load(open(os.path.join(MODELS_FOLDER, "base_decision_tree.pkl"), "rb"))

# %%
train_y_pred = np.floor(model.predict(train_X)).astype(int)
test_y_pred = np.floor(model.predict(test_X)).astype(int)

# %%
model_name = "Decision Tree"
results[("Train", "RMSE")][model_name] = root_mean_squared_error(train_y, train_y_pred)
results[("Train", "MAE")][model_name] = mean_absolute_error(train_y, train_y_pred)
results[("Train", "EVS")][model_name] = explained_variance_score(train_y, train_y_pred)
results[("Train", "R2")][model_name] = r2_score(train_y, train_y_pred)
results[("Test", "RMSE")][model_name] = root_mean_squared_error(test_y, test_y_pred)
results[("Test", "MAE")][model_name] = mean_absolute_error(test_y, test_y_pred)
results[("Test", "EVS")][model_name] = explained_variance_score(test_y, test_y_pred)
results[("Test", "R2")][model_name] = r2_score(test_y, test_y_pred)

# %%
_, ax = plt.subplots(1, 2, figsize=(20, 9))
display = PredictionErrorDisplay(y_true=test_y, y_pred=test_y_pred)
_ = display.plot(
    ax[0], kind="actual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)
_ = display.plot(
    ax[1], kind="residual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)

# %%
_, ax = plt.subplots(1, 1, figsize=(20, 10))
_ = plot_tree(
    model,
    max_depth=4,
    feature_names=train_X.columns,
    filled=True,
    proportion=True,
    rounded=True,
    precision=2,
    fontsize=6,
    ax=ax,
)

# %%
feat_imp = pd.DataFrame(
    {
        "Feature": [x for x in model.feature_names_in_],
        "Importance": [x for x in model.feature_importances_],
    }
)
feat_imp = feat_imp.sort_values("Importance", ascending=False)
_, ax = plt.subplots(1, 1, figsize=(20, 6))
_ = sns.barplot(feat_imp, x="Feature", y="Importance")
_ = plt.title("Feature Importance for Decision Tree Regressor")

# %%
del model

# %% [markdown]
# ## Decision Tree (tuned)

# %%
model = pickle.load(open(os.path.join(MODELS_FOLDER, "tuned_decision_tree.pkl"), "rb"))

# %%
train_y_pred = np.floor(model.predict(train_X)).astype(int)
test_y_pred = np.floor(model.predict(test_X)).astype(int)

# %%
model_name = "Decision Tree (tuned)"
results[("Train", "RMSE")][model_name] = root_mean_squared_error(train_y, train_y_pred)
results[("Train", "MAE")][model_name] = mean_absolute_error(train_y, train_y_pred)
results[("Train", "EVS")][model_name] = explained_variance_score(train_y, train_y_pred)
results[("Train", "R2")][model_name] = r2_score(train_y, train_y_pred)
results[("Test", "RMSE")][model_name] = root_mean_squared_error(test_y, test_y_pred)
results[("Test", "MAE")][model_name] = mean_absolute_error(test_y, test_y_pred)
results[("Test", "EVS")][model_name] = explained_variance_score(test_y, test_y_pred)
results[("Test", "R2")][model_name] = r2_score(test_y, test_y_pred)

# %%
_, ax = plt.subplots(1, 2, figsize=(20, 9))
display = PredictionErrorDisplay(y_true=test_y, y_pred=test_y_pred)
_ = display.plot(
    ax[0], kind="actual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)
_ = display.plot(
    ax[1], kind="residual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)

# %%
_, ax = plt.subplots(1, 1, figsize=(20, 10))
_ = plot_tree(
    model,
    max_depth=4,
    feature_names=train_X.columns,
    filled=True,
    proportion=True,
    rounded=True,
    precision=2,
    fontsize=5,
    ax=ax,
)

# %%
feat_imp = pd.DataFrame(
    {
        "Feature": [x for x in model.feature_names_in_],
        "Importance": [x for x in model.feature_importances_],
    }
)
feat_imp = feat_imp.sort_values("Importance", ascending=False)
_, ax = plt.subplots(1, 1, figsize=(20, 6))
_ = sns.barplot(feat_imp, x="Feature", y="Importance")
_ = plt.title("Feature Importance for Tuned Decision Tree Regressor")

# %%
del model

# %% [markdown]
# ## Random Forest

# %%
model = pickle.load(open(os.path.join(MODELS_FOLDER, "base_random_forest.pkl"), "rb"))

# %%
train_y_pred = np.floor(model.predict(train_X)).astype(int)
test_y_pred = np.floor(model.predict(test_X)).astype(int)

# %%
model_name = "Random Forest"
results[("Train", "RMSE")][model_name] = root_mean_squared_error(train_y, train_y_pred)
results[("Train", "MAE")][model_name] = mean_absolute_error(train_y, train_y_pred)
results[("Train", "EVS")][model_name] = explained_variance_score(train_y, train_y_pred)
results[("Train", "R2")][model_name] = r2_score(train_y, train_y_pred)
results[("Test", "RMSE")][model_name] = root_mean_squared_error(test_y, test_y_pred)
results[("Test", "MAE")][model_name] = mean_absolute_error(test_y, test_y_pred)
results[("Test", "EVS")][model_name] = explained_variance_score(test_y, test_y_pred)
results[("Test", "R2")][model_name] = r2_score(test_y, test_y_pred)

# %%
_, ax = plt.subplots(1, 2, figsize=(20, 9))
display = PredictionErrorDisplay(y_true=test_y, y_pred=test_y_pred)
_ = display.plot(
    ax[0], kind="actual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)
_ = display.plot(
    ax[1], kind="residual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)

# %%
_, ax = plt.subplots(3, 1, figsize=(20, 32))
for i in range(3):
    _ = plot_tree(
        model.estimators_[i * len(model.estimators_) // 3],
        max_depth=4,
        feature_names=train_X.columns,
        filled=True,
        proportion=True,
        rounded=True,
        precision=2,
        fontsize=6,
        ax=ax[i],
    )

# %%
feat_imp = pd.DataFrame(
    {
        "Feature": [x for x in model.feature_names_in_],
        "Importance": [x for x in model.feature_importances_],
    }
)
feat_imp = feat_imp.sort_values("Importance", ascending=False)
_, ax = plt.subplots(1, 1, figsize=(20, 6))
_ = sns.barplot(feat_imp, x="Feature", y="Importance")
_ = plt.title("Feature Importance for Random Forest Regressor")

# %%
del model

# %% [markdown]
# ## Random Forest (tuned)

# %%
model = pickle.load(open(os.path.join(MODELS_FOLDER, "tuned_random_forest.pkl"), "rb"))

# %%
train_y_pred = np.floor(model.predict(train_X)).astype(int)
test_y_pred = np.floor(model.predict(test_X)).astype(int)

# %%
model_name = "Random Forest (tuned)"
results[("Train", "RMSE")][model_name] = root_mean_squared_error(train_y, train_y_pred)
results[("Train", "MAE")][model_name] = mean_absolute_error(train_y, train_y_pred)
results[("Train", "EVS")][model_name] = explained_variance_score(train_y, train_y_pred)
results[("Train", "R2")][model_name] = r2_score(train_y, train_y_pred)
results[("Test", "RMSE")][model_name] = root_mean_squared_error(test_y, test_y_pred)
results[("Test", "MAE")][model_name] = mean_absolute_error(test_y, test_y_pred)
results[("Test", "EVS")][model_name] = explained_variance_score(test_y, test_y_pred)
results[("Test", "R2")][model_name] = r2_score(test_y, test_y_pred)

# %%
_, ax = plt.subplots(1, 2, figsize=(20, 9))
display = PredictionErrorDisplay(y_true=test_y, y_pred=test_y_pred)
_ = display.plot(
    ax[0], kind="actual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)
_ = display.plot(
    ax[1], kind="residual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)

# %%
_, ax = plt.subplots(3, 1, figsize=(20, 32))
for i in range(3):
    _ = plot_tree(
        model.estimators_[i * len(model.estimators_) // 3],
        max_depth=4,
        feature_names=train_X.columns,
        filled=True,
        proportion=True,
        rounded=True,
        precision=2,
        fontsize=6,
        ax=ax[i],
    )

# %%
feat_imp = pd.DataFrame(
    {
        "Feature": [x for x in model.feature_names_in_],
        "Importance": [x for x in model.feature_importances_],
    }
)
feat_imp = feat_imp.sort_values("Importance", ascending=False)
_, ax = plt.subplots(1, 1, figsize=(20, 6))
_ = sns.barplot(feat_imp, x="Feature", y="Importance")
_ = plt.title("Feature Importance for Tuned Random Forest Regressor")

# %%
del model

# %% [markdown]
# ## Gradient Boosted Trees

# %%
model = pickle.load(open(os.path.join(MODELS_FOLDER, "base_gbt.pkl"), "rb"))

# %%
train_y_pred = np.floor(model.predict(train_X)).astype(int)
test_y_pred = np.floor(model.predict(test_X)).astype(int)

# %%
model_name = "Gradient Boosted Trees"
results[("Train", "RMSE")][model_name] = root_mean_squared_error(train_y, train_y_pred)
results[("Train", "MAE")][model_name] = mean_absolute_error(train_y, train_y_pred)
results[("Train", "EVS")][model_name] = explained_variance_score(train_y, train_y_pred)
results[("Train", "R2")][model_name] = r2_score(train_y, train_y_pred)
results[("Test", "RMSE")][model_name] = root_mean_squared_error(test_y, test_y_pred)
results[("Test", "MAE")][model_name] = mean_absolute_error(test_y, test_y_pred)
results[("Test", "EVS")][model_name] = explained_variance_score(test_y, test_y_pred)
results[("Test", "R2")][model_name] = r2_score(test_y, test_y_pred)

# %%
_, ax = plt.subplots(1, 2, figsize=(20, 9))
display = PredictionErrorDisplay(y_true=test_y, y_pred=test_y_pred)
_ = display.plot(
    ax[0], kind="actual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)
_ = display.plot(
    ax[1], kind="residual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)

# %%
_, ax = plt.subplots(3, 1, figsize=(20, 32))
for i in range(3):
    _ = plot_tree(
        model.estimators_[i * len(model.estimators_) // 3][0],
        max_depth=4,
        feature_names=train_X.columns,
        filled=True,
        proportion=True,
        rounded=True,
        precision=2,
        fontsize=9,
        ax=ax[i],
    )

# %%
feat_imp = pd.DataFrame(
    {
        "Feature": [x for x in model.feature_names_in_],
        "Importance": [x for x in model.feature_importances_],
    }
)
feat_imp = feat_imp.sort_values("Importance", ascending=False)
_, ax = plt.subplots(1, 1, figsize=(20, 6))
_ = sns.barplot(feat_imp, x="Feature", y="Importance")
_ = plt.title("Feature Importance for Gradient Boosted Trees Regressor")

# %%
del model

# %% [markdown]
# ## Gradient Boosted Trees (tuned)

# %%
model = pickle.load(open(os.path.join(MODELS_FOLDER, "tuned_gbt.pkl"), "rb"))

# %%
train_y_pred = np.floor(model.predict(train_X)).astype(int)
test_y_pred = np.floor(model.predict(test_X)).astype(int)

# %%
model_name = "Gradient Boosted Trees (tuned)"
results[("Train", "RMSE")][model_name] = root_mean_squared_error(train_y, train_y_pred)
results[("Train", "MAE")][model_name] = mean_absolute_error(train_y, train_y_pred)
results[("Train", "EVS")][model_name] = explained_variance_score(train_y, train_y_pred)
results[("Train", "R2")][model_name] = r2_score(train_y, train_y_pred)
results[("Test", "RMSE")][model_name] = root_mean_squared_error(test_y, test_y_pred)
results[("Test", "MAE")][model_name] = mean_absolute_error(test_y, test_y_pred)
results[("Test", "EVS")][model_name] = explained_variance_score(test_y, test_y_pred)
results[("Test", "R2")][model_name] = r2_score(test_y, test_y_pred)

# %%
_, ax = plt.subplots(1, 2, figsize=(20, 9))
display = PredictionErrorDisplay(y_true=test_y, y_pred=test_y_pred)
_ = display.plot(
    ax[0], kind="actual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)
_ = display.plot(
    ax[1], kind="residual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)

# %%
_, ax = plt.subplots(3, 1, figsize=(20, 32))
for i in range(3):
    _ = plot_tree(
        model.estimators_[i * len(model.estimators_) // 3][0],
        max_depth=4,
        feature_names=train_X.columns,
        filled=True,
        proportion=True,
        rounded=True,
        precision=2,
        fontsize=5,
        ax=ax[i],
    )

# %%
feat_imp = pd.DataFrame(
    {
        "Feature": [x for x in model.feature_names_in_],
        "Importance": [x for x in model.feature_importances_],
    }
)
feat_imp = feat_imp.sort_values("Importance", ascending=False)
_, ax = plt.subplots(1, 1, figsize=(20, 6))
_ = sns.barplot(feat_imp, x="Feature", y="Importance")
_ = plt.title("Feature Importance for Tuned Gradient Boosted Trees Regressor")

# %%
del model

# %% [markdown]
# ## XGBoost

# %%
model = pickle.load(open(os.path.join(MODELS_FOLDER, "base_xgboost.pkl"), "rb"))

# %%
train_y_pred = np.floor(model.predict(train_X)).astype(int)
test_y_pred = np.floor(model.predict(test_X)).astype(int)

# %%
model_name = "XGBoost"
results[("Train", "RMSE")][model_name] = root_mean_squared_error(train_y, train_y_pred)
results[("Train", "MAE")][model_name] = mean_absolute_error(train_y, train_y_pred)
results[("Train", "EVS")][model_name] = explained_variance_score(train_y, train_y_pred)
results[("Train", "R2")][model_name] = r2_score(train_y, train_y_pred)
results[("Test", "RMSE")][model_name] = root_mean_squared_error(test_y, test_y_pred)
results[("Test", "MAE")][model_name] = mean_absolute_error(test_y, test_y_pred)
results[("Test", "EVS")][model_name] = explained_variance_score(test_y, test_y_pred)
results[("Test", "R2")][model_name] = r2_score(test_y, test_y_pred)

# %%
_, ax = plt.subplots(1, 2, figsize=(20, 9))
display = PredictionErrorDisplay(y_true=test_y, y_pred=test_y_pred)
_ = display.plot(
    ax[0], kind="actual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)
_ = display.plot(
    ax[1], kind="residual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)

# %%
feat_imp = pd.DataFrame(
    {
        "Feature": [x for x in model.feature_names_in_],
        "Importance": [x for x in model.feature_importances_],
    }
)
feat_imp = feat_imp.sort_values("Importance", ascending=False)
_, ax = plt.subplots(1, 1, figsize=(20, 6))
_ = sns.barplot(feat_imp, x="Feature", y="Importance")
_ = plt.title("Feature Importance for XGBoost Regressor")

# %%
del model

# %% [markdown]
# ## XGBoost (tuned)

# %%
model = pickle.load(open(os.path.join(MODELS_FOLDER, "tuned_xgboost.pkl"), "rb"))

# %%
train_y_pred = np.floor(model.predict(train_X)).astype(int)
test_y_pred = np.floor(model.predict(test_X)).astype(int)

# %%
model_name = "XGBoost (tuned)"
results[("Train", "RMSE")][model_name] = root_mean_squared_error(train_y, train_y_pred)
results[("Train", "MAE")][model_name] = mean_absolute_error(train_y, train_y_pred)
results[("Train", "EVS")][model_name] = explained_variance_score(train_y, train_y_pred)
results[("Train", "R2")][model_name] = r2_score(train_y, train_y_pred)
results[("Test", "RMSE")][model_name] = root_mean_squared_error(test_y, test_y_pred)
results[("Test", "MAE")][model_name] = mean_absolute_error(test_y, test_y_pred)
results[("Test", "EVS")][model_name] = explained_variance_score(test_y, test_y_pred)
results[("Test", "R2")][model_name] = r2_score(test_y, test_y_pred)

# %%
_, ax = plt.subplots(1, 2, figsize=(20, 9))
display = PredictionErrorDisplay(y_true=test_y, y_pred=test_y_pred)
_ = display.plot(
    ax[0], kind="actual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)
_ = display.plot(
    ax[1], kind="residual_vs_predicted", scatter_kwargs={"linewidth": 0, "s": 1}
)

# %%
feat_imp = pd.DataFrame(
    {
        "Feature": [x for x in model.feature_names_in_],
        "Importance": [x for x in model.feature_importances_],
    }
)
feat_imp = feat_imp.sort_values("Importance", ascending=False)
_, ax = plt.subplots(1, 1, figsize=(20, 6))
_ = sns.barplot(feat_imp, x="Feature", y="Importance")
_ = plt.title("Feature Importance for Tuned XGBoost Regressor")

# %%
del model

# %% [markdown]
# ## Results

# %%
pd.DataFrame(results).sort_values(("Test", "R2"), ascending=False)


