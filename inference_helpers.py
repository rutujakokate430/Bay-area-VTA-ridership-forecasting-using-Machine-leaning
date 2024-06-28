# %%
import polars as pl
import os

# %%
RAW_DATA_FOLDER = "raw_data"
STAGING_DATA_FOLDER = "staging_data"

# %%
ridership = pl.scan_csv(os.path.join(RAW_DATA_FOLDER, "Ridership.csv")).with_columns(
    pl.col("Stop Id").str.replace_all(",", "").cast(pl.UInt16),
)

# %%
ridership.select(["Line", "Direction Number", "Sequence", "Stop Id"]).unique().sort("Stop Id").collect(
    streaming=True
).write_csv(os.path.join(STAGING_DATA_FOLDER, "line_sequence_stop.csv"))

# %%
ridership.select(["Stop Id", "Stop Name"]).unique().sort("Stop Id").collect(
    streaming=True
).write_csv(os.path.join(STAGING_DATA_FOLDER, "stop_names.csv"))


