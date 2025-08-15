from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression


# Function to calculate the normalized distance (nd) between baseline and anti-pattern datasets
def calculate_normalized_distance(baseline_df, pattern_df):
    merged_df = baseline_df.merge(pattern_df, on=['event_op_name', 'load'], suffixes=('_baseline', '_pattern'))

    # Calculating normalized distance
    merged_df['nd_max_response_time'] = 2 * (merged_df['max_response_time_pattern'] / (
                merged_df['max_response_time_pattern'] + merged_df['max_response_time_baseline']))
    merged_df['nd_mean_response_time'] = 2 * (merged_df['mean_response_time_pattern'] / (
                merged_df['mean_response_time_pattern'] + merged_df['mean_response_time_baseline']))
    merged_df['nd_std_response_time'] = 2 * (merged_df['std_response_time_pattern'] / (
                merged_df['std_response_time_pattern'] + merged_df['std_response_time_baseline']))

    merged_df['nd_max_ret'] = 2 * (
                merged_df['max_ret_pattern'] / (merged_df['max_ret_pattern'] + merged_df['max_ret_baseline']))
    merged_df['nd_mean_ret'] = 2 * (
                merged_df['mean_ret_pattern'] / (merged_df['mean_ret_pattern'] + merged_df['mean_ret_baseline']))

    merged_df['nd_max_count'] = 2 * (
                merged_df['max_count_pattern'] / (merged_df['max_count_pattern'] + merged_df['max_count_baseline']))
    merged_df['nd_mean_count'] = 2 * (
                merged_df['mean_count_pattern'] / (merged_df['mean_count_pattern'] + merged_df['mean_count_baseline']))

    merged_df['nd_total_event_error_count'] = 2 * (merged_df['total_event_error_count_pattern'] / (
                merged_df['total_event_error_count_pattern'] + merged_df['total_event_error_count_baseline']))
    print(merged_df)
    return merged_df

def calculate_slope(df, metrics):
    slopes = {}
    for metric in metrics:
        slopes[f'slope_{metric}'] = []

    # Group by event_op_name and calculate the slope for each metric
    for event_op_name, group in df.groupby('event_op_name'):
        X = group['load'].values.reshape(-1, 1)
        for metric in metrics:
            y = group[metric].values
            model = LinearRegression().fit(X, y)
            slopes[f'slope_{metric}'].append(model.coef_[0])

    slope_df = pd.DataFrame(slopes)
    slope_df['event_op_name'] = list(df['event_op_name'].unique())

    return slope_df

def data_preparation(blob_nd_df, est_nd_df, baseline_slope_df, blob_slope_df, est_slope_df):
    # Add labels
    blob_nd_df['label'] = 1
    est_nd_df['label'] = 2

    # For baseline, normalized distances should be 1, so we create a similar structure with label 0
    baseline_nd_df = blob_nd_df.copy()
    for col in baseline_nd_df.columns:
        if 'nd' in col:
            baseline_nd_df[col] = 1
    baseline_nd_df['label'] = 0

    # Merge the slopes with the normalized distances
    blob_nd_slope_df = pd.merge(blob_nd_df, blob_slope_df, on='event_op_name')
    est_nd_slope_df = pd.merge(est_nd_df, est_slope_df, on='event_op_name')
    baseline_nd_slope_df = pd.merge(baseline_nd_df, baseline_slope_df, on='event_op_name')

    # Combine all datasets
    combined_nd_slope_df = pd.concat([baseline_nd_slope_df, blob_nd_slope_df, est_nd_slope_df])

    # Select features (normalized distances and slopes) and labels
    features_nd_slope = combined_nd_slope_df.filter(regex='nd_|slope_').fillna(0)
    labels_nd_slope = combined_nd_slope_df['label']

    return features_nd_slope, labels_nd_slope


def model_training(features_nd_slope, labels_nd_slope):

    # Split data into training and testing sets
    X_train_nd_slope, X_test_nd_slope, y_train_nd_slope, y_test_nd_slope = train_test_split(features_nd_slope,
                                                                                            labels_nd_slope,
                                                                                            test_size=0.3,
                                                                                            random_state=42)

    # Train a RandomForestClassifier
    model_nd_slope = RandomForestClassifier(random_state=42)
    model_nd_slope.fit(X_train_nd_slope, y_train_nd_slope)

    # Predict on the test set
    y_pred_nd_slope = model_nd_slope.predict(X_test_nd_slope)

    # Evaluate the model
    report_nd_slope = classification_report(y_test_nd_slope, y_pred_nd_slope)
    print(y_pred_nd_slope)
    print(report_nd_slope)


if __name__ == "__main__":
    baseline_df = pd.read_csv(sys.argv[1])
    blob_df = pd.read_csv(sys.argv[2])
    est_df = pd.read_csv(sys.argv[3])

    # Handle NaN values by filling them with 0
    baseline_df.fillna(0, inplace=True)
    blob_df.fillna(0, inplace=True)
    est_df.fillna(0, inplace=True)

    # Calculate normalized distances for both anti-patterns
    blob_nd_df = calculate_normalized_distance(baseline_df, blob_df)
    est_nd_df = calculate_normalized_distance(baseline_df, est_df)

    # Define metrics to calculate slopes for
    metrics = [
        'max_response_time',
        'mean_response_time',
        'std_response_time',
        'max_ret',
        'mean_ret',
        'max_count',
        'mean_count',
        'total_event_error_count'
    ]

    # Calculate slopes for the baseline, blob, and EST datasets
    baseline_slope_df = calculate_slope(baseline_df, metrics)
    blob_slope_df = calculate_slope(blob_df, metrics)
    est_slope_df = calculate_slope(est_df, metrics)


    features_nd_slope, labels_nd_slope = data_preparation(blob_nd_df, est_nd_df, baseline_slope_df, blob_slope_df, est_slope_df)
    model_training(features_nd_slope, labels_nd_slope)

