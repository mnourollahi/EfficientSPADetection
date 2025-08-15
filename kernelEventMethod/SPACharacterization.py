import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


# Function to calculate slope for each event type
def calculate_slope_by_event(anti_pattern_df, feature, event_op_name):
    filtered_df = anti_pattern_df[anti_pattern_df['event_op_name'] == event_op_name]
    loads = sorted(filtered_df['load'].unique())
    feature_means = []

    for load in loads:
        feature_mean = filtered_df[filtered_df['load'] == load][feature].mean()
        feature_means.append(feature_mean)

    # Perform linear regression (Load vs Performance Metric)
    X = np.array(loads).reshape(-1, 1)  # Load is the independent variable (x-axis)
    Y = np.array(feature_means)  # Performance metric is the dependent variable (y-axis)

    # Handle any NaN or infinite values in Y
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    if len(Y) > 1:  # Need at least two points to fit the regression line
        reg = LinearRegression().fit(X, Y)
        return reg.coef_[0]  # Return the slope of the regression line
    else:
        return 0  # Return 0 if not enough data to calculate slope


# Function to calculate per-load normalized distance for each event type
def calculate_per_load_normalized_distance_by_event(baseline_df, anti_pattern_df, feature, event_op_name):
    filtered_baseline_df = baseline_df[baseline_df['event_op_name'] == event_op_name]
    filtered_anti_pattern_df = anti_pattern_df[anti_pattern_df['event_op_name'] == event_op_name]

    loads = sorted(filtered_baseline_df['load'].unique())
    normalized_distances = []

    for load in loads:
        baseline_values = filtered_baseline_df[filtered_baseline_df['load'] == load][feature].mean()
        anti_pattern_values = filtered_anti_pattern_df[filtered_anti_pattern_df['load'] == load][feature].mean()

        # Handle NaN values by replacing them with 0
        if np.isnan(baseline_values):
            baseline_values = 0
        if np.isnan(anti_pattern_values):
            anti_pattern_values = 0

        # Avoid division by zero and infinity
        if baseline_values == 0 and anti_pattern_values == 0:
            nd = 0
        else:
            nd = 2 * anti_pattern_values / (anti_pattern_values + baseline_values)

        normalized_distances.append(nd)

    return loads, normalized_distances


# Function to generate plots for each event type
def plot_event_type_metrics(baseline_df, blob_df, est_df, event_op_name, features):
    plt.figure(figsize=(10, 6))

    for i, feature in enumerate(features):
        # Calculate slope and normalized distance for blob
        blob_slope = calculate_slope_by_event(blob_df, feature, event_op_name)
        blob_loads, blob_nds = calculate_per_load_normalized_distance_by_event(baseline_df, blob_df, feature,
                                                                               event_op_name)

        # Calculate slope and normalized distance for est
        est_slope = calculate_slope_by_event(est_df, feature, event_op_name)
        est_loads, est_nds = calculate_per_load_normalized_distance_by_event(baseline_df, est_df, feature,
                                                                             event_op_name)

        # Plot Blob data
        plt.scatter(blob_nds, [blob_slope] * len(blob_nds), label=f'Blob {feature}', marker='o')

        # Plot EST data
        plt.scatter(est_nds, [est_slope] * len(est_nds), label=f'EST {feature}', marker='x')

    plt.xlabel('Normalized Distance')
    plt.ylabel('Slope')
    plt.title(f'Slope vs Normalized Distance for Event: {event_op_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(str(event_op_name), dpi=300, bbox_inches='tight')


# Function to calculate slope of the regression line (Load vs Performance Metric)
def calculate_slope(anti_pattern_df, feature):
    loads = sorted(anti_pattern_df['load'].unique())
    feature_means = []

    for load in loads:
        anti_pattern_values = anti_pattern_df[anti_pattern_df['load'] == load][feature].mean()
        feature_means.append(anti_pattern_values)

    # Perform linear regression (Load vs Performance Metric)
    X = np.array(loads).reshape(-1, 1)  # Load is the independent variable (x-axis)
    Y = np.array(feature_means)  # Performance metric is the dependent variable (y-axis)

    # Handle any NaN or infinite values in Y
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    if len(Y) > 1:  # Need at least two points to fit the regression line
        reg = LinearRegression().fit(X, Y)
        return reg.coef_[0]  # Return the slope of the regression line
    else:
        return 0  # Return 0 if not enough data to calculate slope


# Function to calculate normalized distance
def calculate_normalized_distance(baseline_df, anti_pattern_df, feature):
    loads = sorted(baseline_df['load'].unique())
    normalized_distances = []

    for load in loads:
        baseline_values = baseline_df[baseline_df['load'] == load][feature].mean()
        anti_pattern_values = anti_pattern_df[anti_pattern_df['load'] == load][feature].mean()

        # Handle NaN values by replacing them with 0
        if np.isnan(baseline_values):
            baseline_values = 0
        if np.isnan(anti_pattern_values):
            anti_pattern_values = 0

        # Avoid division by zero and infinity
        if baseline_values == 0 and anti_pattern_values == 0:
            nd = 0
        else:
            nd = 2 * anti_pattern_values / (anti_pattern_values + baseline_values)

        normalized_distances.append(nd)

    # Handle any NaN or infinite values in normalized distances
    normalized_distances = np.nan_to_num(normalized_distances, nan=0.0, posinf=0.0, neginf=0.0)

    return np.mean(normalized_distances)  # Return average normalized distance across all loads

# Function to calculate per-load normalized distance
def calculate_per_load_normalized_distance(baseline_df, anti_pattern_df, feature):
    loads = sorted(baseline_df['load'].unique())
    normalized_distances = []

    for load in loads:
        baseline_values = baseline_df[baseline_df['load'] == load][feature].mean()
        anti_pattern_values = anti_pattern_df[anti_pattern_df['load'] == load][feature].mean()

        # Handle NaN values by replacing them with 0
        if np.isnan(baseline_values):
            baseline_values = 0
        if np.isnan(anti_pattern_values):
            anti_pattern_values = 0

        # Avoid division by zero and infinity
        if baseline_values == 0 and anti_pattern_values == 0:
            nd = 0
        else:
            nd = 2 * anti_pattern_values / (anti_pattern_values + baseline_values)

        normalized_distances.append(nd)

    return loads, normalized_distances


import matplotlib.pyplot as plt


# Function to plot metrics for each event type
def plot_metrics_per_event_type(df, features):
    event_op_names = df['event_op_name'].unique()

    # Loop over each event type (event_op_name)
    for event_op_name in event_op_names:
        event_data = df[df['event_op_name'] == event_op_name]

        plt.figure(figsize=(10, 6))

        # Plot each feature for the current event type
        for feature in features:
            plt.plot(event_data['load'], event_data[feature], marker='o', label=feature)

        plt.title(f'Metrics vs Load for Event Type: {event_op_name}')
        plt.xlabel('Load')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    # Load the datasets
    baseline_df = pd.read_csv("baseline_not_agg.txt")
    blob_df = pd.read_csv("blob_not_agg.txt")
    est_df = pd.read_csv("est_not_agg.txt")

    # List of features to process
    features = ['response_time', 'ret', 'count', 'event_error_count']

    # Get unique event types from the dataset
    event_op_names = baseline_df['event_op_name'].unique()

    # Plot for each event type
    for event_op_name in event_op_names:
        plot_event_type_metrics(baseline_df, blob_df, est_df, event_op_name, features)

    # Assuming df is the dataframe with your data
    plot_metrics_per_event_type(df, features)



# Entry point
if __name__ == "__main__":
    main()

