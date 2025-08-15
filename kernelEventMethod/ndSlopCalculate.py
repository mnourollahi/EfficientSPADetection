import numpy as np
import pandas as pd
import os
import sys
from sklearn.linear_model import LinearRegression


def pad_dataframe(df, max_rows):
    padded_df = df.copy()

    # Group by event_op_name and load
    grouped = df.groupby(['event_op_name', 'load'])

    # List to hold all dataframes including padding rows
    all_dfs = [padded_df]

    # Iterate over each group and pad
    for (event_op_name, load), group in grouped:
        # Calculate how many rows to add
        padding_needed = max_rows - len(group)

        if padding_needed > 0:
            # Create padding rows with NaN values (except for the identifying columns)
            padding_rows = pd.DataFrame({
                'event_op_name': [event_op_name] * padding_needed,
                'cur_ts': [pd.NaT] * padding_needed,
                'load': [load] * padding_needed,
                'max_response_time': [None] * padding_needed,
                'mean_response_time': [None] * padding_needed,
                'std_response_time': [None] * padding_needed,
                'max_ret': [None] * padding_needed,
                'mean_ret': [None] * padding_needed,
                'max_count': [None] * padding_needed,
                'mean_count': [None] * padding_needed,
                'total_event_error_count': [None] * padding_needed
            })

            # Add the padding rows to the list of dataframes
            all_dfs.append(padding_rows)

    # Concatenate all dataframes in the list
    padded_df = pd.concat(all_dfs, ignore_index=True)

    return padded_df

# Adjusted function to calculate normalized distance and handle division by zero for entire Series
def calculate_normalized_distance_series(anti_pattern_series, baseline_series):
    sum_series = anti_pattern_series + baseline_series
    return 2 * (anti_pattern_series - baseline_series) / sum_series.replace(0, np.nan)

# Function to calculate the slope of linear regression over time (cur_ts)
def calculate_slope(df, x_col, y_col):
    # Ensure no NaN values
    df = df.dropna(subset=[x_col, y_col])

    if len(df) < 2:
        return np.nan

    X = np.array(df[x_col].astype(float)).reshape(-1, 1)
    y = df[y_col].astype(float)

    model = LinearRegression()
    model.fit(X, y)

    return model.coef_[0]


import pandas as pd


# Custom function to convert the cur_ts format "-1 days 23:39:00" into a numerical value in seconds
def convert_cur_ts_to_seconds(cur_ts):
    # If cur_ts is NaN, return NaN
    if pd.isna(cur_ts):
        return np.nan

    # Split the string into days and time
    parts = cur_ts.split(' ')

    # Extract the time part and convert to seconds
    time_part = parts[-1]
    h, m, s = map(int, time_part.split(':'))
    total_seconds = h * 3600 + m * 60 + s

    # Adjust for the days part
    if len(parts) == 3:
        days = int(parts[0].replace('days', '').strip())
        total_seconds += days * 86400

    return total_seconds

if __name__ == "__main__":
    baseline_processed = pd.read_csv(sys.argv[1])
    blob_processed = pd.read_csv(sys.argv[2])
    est_processed = pd.read_csv(sys.argv[3])

    # Determine the maximum number of rows for each (event_op_name, load) pair across all datasets
    max_rows_per_pair = max(
        baseline_processed.groupby(['event_op_name', 'load']).size().max(),
        blob_processed.groupby(['event_op_name', 'load']).size().max(),
        est_processed.groupby(['event_op_name', 'load']).size().max()
    )

    # Pad each dataset
    padded_baseline_df = pad_dataframe(baseline_processed, max_rows_per_pair)
    padded_blob_df = pad_dataframe(blob_processed, max_rows_per_pair)
    padded_est_df = pad_dataframe(est_processed, max_rows_per_pair)

    # Check the results
    padded_baseline_df.groupby(['event_op_name', 'load']).size().head(), \
        padded_blob_df.groupby(['event_op_name', 'load']).size().head(), \
        padded_est_df.groupby(['event_op_name', 'load']).size().head()

    # Merging datasets on common keys: event_op_name, cur_ts, and load
    merged_df = padded_baseline_df.merge(
        padded_blob_df, on=['event_op_name', 'cur_ts', 'load'], suffixes=('_baseline', '_blob')
    ).merge(
        padded_est_df, on=['event_op_name', 'cur_ts', 'load'], suffixes=('', '_est')
    )

    # Display the first few rows of the merged dataset
    print(merged_df.head())

    # Assuming `merged_df` is your DataFrame containing the `cur_ts` column
    merged_df['cur_ts_numeric'] = merged_df['cur_ts'].apply(convert_cur_ts_to_seconds)
    # Redefine the metrics variable
    metrics = ['max_response_time', 'mean_response_time', 'std_response_time', 'max_ret', 'mean_ret', 'max_count',
               'mean_count']

    # Recalculate normalized distances with the adjusted function
    for metric in metrics:
        merged_df[f'normalized_distance_blob_{metric}'] = calculate_normalized_distance_series(
            merged_df[f'{metric}_blob'], merged_df[f'{metric}_baseline']
        )
        merged_df[f'normalized_distance_est_{metric}'] = calculate_normalized_distance_series(
            merged_df[f'{metric}'], merged_df[f'{metric}_baseline']
        ).fillna(0)  # Fill any NaN values resulting from division by zero with 0

    # Recalculate slopes of linear regression for the metrics over time for the "est" dataset
    for metric in metrics:
        merged_df[f'slope_est_{metric}'] = merged_df.groupby(['event_op_name', 'load']).apply(
            lambda x: calculate_slope(x, 'cur_ts_numeric', f'{metric}')
        ).reset_index(drop=True)

    # Display the first few rows of the merged dataframe with new calculations
    print(merged_df.head())
    merged_df.to_csv('merged_df.txt', index=False)
