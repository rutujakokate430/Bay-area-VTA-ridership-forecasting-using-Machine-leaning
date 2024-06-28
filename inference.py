# %% [markdown]
# # Inference Script for VTA Ridership Prediction
# 
# ***For predictions using Weather Data***

# %% [markdown]
# ## Import packages

# %%
import pandas as pd
import numpy as np
import pickle
import os

# %%
STAGING_DATA_FOLDER = "staging_data"
MODEL_DATA_FOLDER = "models"
model = pickle.load(open(os.path.join(MODEL_DATA_FOLDER, "final_model.pkl"), "rb"))
line_sequence_stop = pd.read_csv(os.path.join(STAGING_DATA_FOLDER, "line_sequence_stop.csv"))
stop_names = pd.read_csv(os.path.join(STAGING_DATA_FOLDER, "stop_names.csv"))
stops = pd.read_csv(os.path.join(STAGING_DATA_FOLDER, "stops.csv"))

# %% [markdown]
# ## User Input here
# 
# - Enter stop name in small caps
# - TMAX is maximum temperature on that day in Fahrenheit
# - TMIN is minimum temperature on that day in Fahrenheit
# - PRCP is precipitation in inches

# %%
INPUT_DATE = "2018-01-25"
INPUT_HOLIDAY = False
INPUT_SPECIAL = False
INPUT_STOP_NAME = "san antonio station"
INPUT_TMAX = 60
INPUT_TMIN = 50
INPUT_PRCP = 0.5

# %%
INPUT_TMAX = (INPUT_TMAX - 32) * 5 / 9
INPUT_TMIN = (INPUT_TMIN - 32) * 5 / 9
INPUT_PRCP = INPUT_PRCP * 2.54

# %%
def determine_service(month, day, weekday, holiday, special):
    # July 4th is considered regardless of the weekday or holiday status unless it is special
    if month == 7 and day == 4:
        return 4

    # Special days handling
    if special:
        if weekday in range(5):  # Monday to Friday
            return 5
        elif weekday == 5:  # Saturday
            return 6
        elif weekday == 6 or holiday:  # Sunday or holiday
            return 7

    # Regular days handling
    if holiday:
        return 3  # Sunday/Holiday mapping
    if weekday == 5:
        return 2  # Saturday mapping
    if weekday == 6:
        return 3  # Sunday mapping

    # Default to weekday if no other conditions are met
    return 1

date = pd.to_datetime(INPUT_DATE).date()
year = pd.to_datetime(INPUT_DATE).year
month = pd.to_datetime(INPUT_DATE).month
day = pd.to_datetime(INPUT_DATE).day_of_year
weekday = pd.to_datetime(INPUT_DATE).weekday
service = determine_service(month, day, weekday, INPUT_HOLIDAY, INPUT_SPECIAL)
date_df = pd.DataFrame({"Year": [year], "Day": [day], "Service": [service], "Date": [date]})
date_df

# %%
climate_df = pd.DataFrame({"Tmax": [INPUT_TMAX], "Tmin": [INPUT_TMIN], "Prcp": [INPUT_PRCP]})
climate_df

# %%
stop_names = stop_names[stop_names["Stop Name"].str.contains(INPUT_STOP_NAME.upper())]
stop_names

# %% [markdown]
# ## User Input here

# %%
INPUT_STOP_IDS = [4747, 4774]

# %%
filtered_stops = stops[stops["Stop Id"].isin(INPUT_STOP_IDS)]
input_df = stop_names.merge(filtered_stops, how="inner", on="Stop Id")
input_df = input_df.merge(line_sequence_stop, on="Stop Id", how="inner").sort_values(
    ["Stop Id", "Line", "Direction Number"]
)
input_df = input_df.merge(date_df, how="cross")
input_df = input_df.merge(climate_df, how="cross")
input_df[
    [
        "Day",
        "Line",
        "Service",
        "Direction Number",
        "Sequence",
        "Latitude",
        "Longitude",
        "Tmax",
        "Tmin",
        "Prcp",
    ]
]

# %%
predictions = model.predict(
    input_df[
        [
            "Day",
            "Line",
            "Service",
            "Direction Number",
            "Sequence",
            "Latitude",
            "Longitude",
            "Tmax",
            "Tmin",
            "Prcp",
        ]
    ]
)

# %%
pred_df = pd.DataFrame({"On": predictions}).apply(np.floor)
output_df = (
    pd.concat(
        [
            input_df[["Date", "Stop Name", "Line", "Service", "Direction Number"]],
            pred_df,
        ],
        axis=1,
    )
    .groupby(["Date", "Stop Name", "Line", "Service", "Direction Number"])
    .mean()
    .apply(np.floor)
    .astype(int)
    .reset_index()
    .sort_values(["Date", "Line", "Service", "Direction Number"])
)
print(output_df.to_string(index=False))


