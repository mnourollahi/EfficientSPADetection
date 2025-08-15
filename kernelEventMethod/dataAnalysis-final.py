import numpy as np
import pandas as pd
import os
import sys
from sklearn.linear_model import LinearRegression


def load_data(folder_path):
    dataframes = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            # Load the dataset from the CSV file
            df = pd.read_csv(file_path)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Function to safely convert timestamps assuming they are in nanoseconds
def safe_convert_to_datetime_ns(timestamps):
    timestamps = pd.to_numeric(timestamps, errors='coerce')
    return pd.to_datetime(timestamps, unit='ns', errors='coerce')

# Function to process each dataset
def process_trace_data(df):
    # Extract the first timestamp from event_cur_ts list
    df['cur_ts'] = df['event_cur_ts'].apply(lambda x: x.strip('[]').split(',')[0] if pd.notnull(x) else None)

    # Safely convert cur_ts to datetime, assuming nanoseconds
    df['cur_ts'] = safe_convert_to_datetime_ns(df['cur_ts'])

    # Calculate delta as the difference between each row's cur_ts and the next row's cur_ts
    df['delta'] = df['cur_ts'].diff().shift(-1).fillna(pd.Timedelta(0))

    # Convert delta to seconds and reset cur_ts to start from 0 and increment with delta
    df['delta'] = df['delta'].dt.total_seconds()
    df['cur_ts'] = df['delta'].cumsum()
    print(df.head(5))
    # Flatten the event_op_name_ids, event_response_times, and event_rets into rows
    processed_data = []
    for index, row in df.iterrows():
        cur_ts = row['cur_ts']
        load = row['load']
        event_response_times = eval(row['event_response_times']) if isinstance(row['event_response_times'], str) else row['event_response_times']
        event_rets = eval(row['event_rets']) if isinstance(row['event_rets'], str) else row['event_rets']
        event_op_name_ids = eval(row['event_op_names_ids']) if isinstance(row['event_op_names_ids'], str) else row['event_op_names_ids']

        # Ensure event_op_name_ids is treated as a list of integers
        if not isinstance(event_op_name_ids, list):
            event_op_name_ids = [event_op_name_ids] if pd.notna(event_op_name_ids) else []



        # Count occurrences of each event_op_name_id category in the request
        event_op_name_id_counts = {event_id: event_op_name_ids.count(event_id) for event_id in set(event_op_name_ids)}

        # Process response times, ret values, and counts for each event_op_name
        for op_name, response_time, ret in zip(event_op_name_ids, event_response_times, event_rets):
            # Initialize event_error_count for this row
            event_error_count = 0
            # Replace None with 0 for ret
            if ret is None:
                ret = 0
            # Increment event_error_count if ret is less than 0
            if ret < 0:
                event_error_count += 1
                ret = 0  # Replace ret with 0 if it is less than 0

            processed_data.append({
                'cur_ts': cur_ts,
                'load': load,
                'event_op_name': op_name,
                'response_time': response_time,
                'ret': ret,
                'count': event_op_name_id_counts[op_name],  # Add the count for this event_op_name
                'event_error_count': event_error_count  # Add the event_error_count to the data
            })

    processed_df = pd.DataFrame(processed_data)

    # Verify if 'cur_ts' column exists
    if 'cur_ts' in processed_df.columns:
        # Convert cur_ts to TimedeltaIndex for grouping
        processed_df.index = pd.to_timedelta(processed_df['cur_ts'], unit='s')
    else:
        raise KeyError("The 'cur_ts' column is missing in the processed data.")

    # # Create 10-second time windows and aggregate max and mean metrics
    aggregated_df = processed_df.groupby(['event_op_name', pd.Grouper(freq='300S')]).agg(
        load=('load', 'first'),
        max_response_time=('response_time', 'max'),
        mean_response_time=('response_time', 'mean'),
        std_response_time=('response_time', 'std'),  # Calculate standard deviation for baseline
        max_ret=('ret', 'max'),
        mean_ret=('ret', 'mean'),
        max_count=('count', 'max'),
        mean_count=('count', 'mean'),
        total_event_error_count=('event_error_count', 'sum')# Aggregate event_error_count
    ).reset_index()

    # # Replace the seconds and microseconds with 0, keeping only hours and minutes
    aggregated_df['cur_ts'] = pd.to_timedelta(aggregated_df['cur_ts'], unit='s').apply(
        lambda x: f"{x.components.days} days {x.components.hours:02}:{x.components.minutes:02}:00")

    # return processed_df
    return aggregated_df, processed_df



# Define a function to pad a DataFrame so that each (event_op_name, load) pair has the same number of rows
def pad_dataframe(df, max_rows):
    padded_df = df.copy()

    # Group by event_op_name and load
    grouped = df.groupby(['event_op_name', 'load'])

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

            # Append padding rows to the group
            padded_df = padded_df.append(padding_rows, ignore_index=True)

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

# Custom function to convert the cur_ts format "-1 days 23:39:00" into a numerical value
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
        days = int(parts[0])
        total_seconds += days * 86400

    return total_seconds

if __name__ == "__main__":
    # Paths to baseline and antipattern folders
    baseline_folder = sys.argv[1]
    blob_folder = sys.argv[2]
    est_folder = sys.argv[3]
    # Load data
    baseline_df = load_data(baseline_folder)
    blob_df = load_data(blob_folder)
    est_df = load_data(est_folder)

    # Combine datasets
    est_df['label'] = 2  # est anti-pattern
    blob_df['label'] = 1  # Blob anti-pattern
    baseline_df['label'] = 0  # Baseline

    # Process each dataset

    # Process the baseline dataset with the new baseline calculation
    baseline_processed, baseline_not_agg = process_trace_data(baseline_df)
    baseline_processed['baseline'] = baseline_processed['mean_response_time'] + 3 * baseline_processed[
        'std_response_time']
    baseline_processed.to_csv('baseline_processed.txt', index=False)
    baseline_not_agg.to_csv('baseline_not_agg.txt', index=False)

    # Process the blob and est datasets as before and add prefixes
    blob_processed, blob_not_agg = process_trace_data(blob_df)
    blob_processed.to_csv('blob_processed.txt', index=False)
    blob_not_agg.to_csv('excessive_not_agg.txt', index=False)
    est_processed, est_not_agg = process_trace_data(est_df)
    est_processed.to_csv('est_processed.txt', index=False)
    est_not_agg.to_csv('est_not_agg.txt', index=False)

    # Determine the maximum number of rows for each (event_op_name, load) pair across all datasets
    max_rows_per_pair = max(
        baseline_processed.groupby(['event_op_name', 'load']).size().max(),
        blob_processed.groupby(['event_op_name', 'load']).size().max(),
        est_processed.groupby(['event_op_name', 'load']).size().max()
    )
    #
    # # Pad each dataset
    padded_baseline_df = pad_dataframe(baseline_processed, max_rows_per_pair)
    padded_blob_df = pad_dataframe(blob_processed, max_rows_per_pair)
    padded_est_df = pad_dataframe(est_processed, max_rows_per_pair)
    #
    # # Check the results
    padded_baseline_df.groupby(['event_op_name', 'load']).size().head(), \
        padded_blob_df.groupby(['event_op_name', 'load']).size().head(), \
        padded_est_df.groupby(['event_op_name', 'load']).size().head()

    # # Merging datasets on common keys: event_op_name, cur_ts, and load
    merged_df = padded_baseline_df.merge(
        padded_blob_df, on=['event_op_name', 'cur_ts', 'load'], suffixes=('_baseline', '_blob')
    ).merge(
        padded_est_df, on=['event_op_name', 'cur_ts', 'load'], suffixes=('', '_est')
    )
    #
    # # Display the first few rows of the merged dataset
    merged_df.head()
    #
    # Convert cur_ts to numerical values for regression analysis
    merged_df['cur_ts_numeric'] = pd.to_datetime(merged_df['cur_ts']).astype(int) / 10 ** 9
    #
    # # Redefine the metrics variable
    metrics = ['max_response_time', 'mean_response_time', 'std_response_time', 'max_ret', 'mean_ret', 'max_count',
               'mean_count']
    #
    # # Recalculate normalized distances with the adjusted function
    for metric in metrics:
        merged_df[f'normalized_distance_blob_{metric}'] = calculate_normalized_distance_series(
            merged_df[f'{metric}_blob'], merged_df[f'{metric}_baseline']
        )
        merged_df[f'normalized_distance_est_{metric}'] = calculate_normalized_distance_series(
            merged_df[f'{metric}'], merged_df[f'{metric}_baseline']
        ).fillna(0)  # Fill any NaN values resulting from division by zero with 0
    #
    # # Recalculate slopes of linear regression for the metrics over time for the "est" dataset
    for metric in metrics:
        merged_df[f'slope_est_{metric}'] = merged_df.groupby(['event_op_name', 'load']).apply(
            lambda x: calculate_slope(x, 'cur_ts_numeric', f'{metric}')
        ).reset_index(drop=True)
    #
    # # Display the first few rows of the merged dataframe with new calculations
    merged_df.head()
