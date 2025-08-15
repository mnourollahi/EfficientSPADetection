import sys
import pandas as pd
import os
import numpy as np
from sklearn.metrics import pairwise_distances
import Levenshtein as lev  # You can install using 'pip install python-Levenshtein'
import pandas as pd
import networkx as nx
from collections import Counter
from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE


def lev_distance(seq1, seq2):
    return lev.distance(' '.join(seq1), ' '.join(seq2))

def calculate_distances(trace_sequences):
    n = len(trace_sequences)
    distance_matrix = np.zeros((n, n))

    # Function to compute distances between all pairs in parallel
    def compute_row(i):
        row = np.zeros(n)
        for j in range(i + 1, n):  # Only compute upper triangle, i < j
            row[j] = lev_distance(trace_sequences[i], trace_sequences[j])
        return i, row

    # Use ProcessPoolExecutor to parallelize the calculation of distance matrix rows
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(compute_row, range(n)))

    # Update the distance matrix with results
    for i, row in results:
        distance_matrix[i, i + 1:] = row[i + 1:]  # Upper triangle
        distance_matrix[i + 1:, i] = row[i + 1:]  # Symmetry: copy to lower triangle

    return distance_matrix

def classifiy_traces(df):
    # Extract sequences of operations for each trace
    trace_sequences = df.groupby('trace_id')['op_name'].apply(list).tolist()

    # Compute the distance matrix in parallel
    distance_matrix = calculate_distances(trace_sequences)

    # Use DBSCAN to cluster traces based on sequence similarity
    dbscan = DBSCAN(metric='precomputed', eps=5, min_samples=2)
    trace_labels = dbscan.fit_predict(distance_matrix)

    # Add the cluster labels back to the original DataFrame
    df['trace_type'] = pd.Series(trace_labels, index=df['trace_id'].unique())

    # Display the cluster labels for each trace
    print(df[['trace_id', 'trace_type']].drop_duplicates())

    # Assuming trace_sequences is already computed
    # Run TSNE to reduce the distance matrix to 2 dimensions for visualization
    tsne = TSNE(n_components=2, metric='precomputed')
    trace_embeddings = tsne.fit_transform(distance_matrix)

    # Visualize the trace clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(trace_embeddings[:, 0], trace_embeddings[:, 1], c=trace_labels, cmap='viridis')
    plt.colorbar(label="Cluster Label")
    plt.title('t-SNE Visualization of Trace Clusters')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    return df

def extract_features(df):
    # Initialize a list to store features for each trace
    trace_features_list = []

    # Group the DataFrame by 'trace_id' to process each trace individually
    for trace_id, trace_df in df.groupby('trace_id'):
        # Sort the trace events chronologically
        trace_df = trace_df.sort_values(by='cur_ts')

        # Extract basic features
        total_response_time = trace_df['response_time'].sum()
        max_response_time = trace_df['response_time'].max()
        mean_response_time = trace_df['response_time'].mean()
        total_operations = trace_df['op_name'].count()
        total_ret = trace_df['ret'].sum()
        max_ret = trace_df['ret'].max()
        mean_ret = trace_df['ret'].mean()

        # Build the service call graph for the trace
        G = nx.DiGraph()
        services = trace_df['service_name'].tolist()

        # Loop through the trace events to build edges between services
        for i in range(len(services) - 1):
            from_service = services[i]
            to_service = services[i + 1]
            if from_service != to_service:
                if G.has_edge(from_service, to_service):
                    G[from_service][to_service]['weight'] += 1
                else:
                    G.add_edge(from_service, to_service, weight=1)

        # Total number of service calls (sum of edge weights)
        total_service_calls = sum([d['weight'] for u, v, d in G.edges(data=True)])

        # Pairwise service call counts
        pairwise_calls = '; '.join([f"{u}->{v}:{d['weight']}" for u, v, d in G.edges(data=True)])

        # Get the trace_type assigned during clustering
        trace_type = trace_df['trace_type'].iloc[0]

        # Collect all features into a dictionary
        trace_features = {
            'trace_id': trace_id,
            'trace_type': trace_type,
            'total_response_time': total_response_time,
            'max_response_time': max_response_time,
            'mean_response_time': mean_response_time,
            'total_operations': total_operations,
            'total_service_calls': total_service_calls,
            'pairwise_service_calls': pairwise_calls,
            'total_ret': total_ret,
            'max_ret': max_ret,
            'mean_ret': mean_ret,
        }

        # Append the features to the list
        trace_features_list.append(trace_features)

    # Convert the list of dictionaries to a DataFrame
    trace_features_df = pd.DataFrame(trace_features_list)

    # Display the features for each trace
    print("Features Extracted for Each Trace:")
    print(trace_features_df.head())

    # Visualize the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Service Call Graph for a Trace")
    plt.show()

    # Visualize the distribution of total service calls and total operations
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(trace_features_df['total_service_calls'], kde=True, color='purple')
    plt.title('Distribution of Total Service Calls')

    plt.subplot(1, 2, 2)
    sns.histplot(trace_features_df['total_operations'], kde=True, color='orange')
    plt.title('Distribution of Total Operations')

    plt.tight_layout()
    plt.show()

    return trace_features_df

def aggregate_features(trace_features_df):
    # Group the trace_features_df by 'trace_type' and compute aggregate statistics
    category_features = trace_features_df.groupby('trace_type').agg({
        'trace_id': 'count',  # Number of traces in the category
        'total_response_time': ['mean', 'max'],
        'max_response_time': 'mean',
        'mean_response_time': 'mean',
        'total_operations': 'mean',
        'total_service_calls': 'mean',
        'total_ret': ['mean', 'max'],
        'max_ret': 'mean',
        'mean_ret': 'mean',
    }).reset_index()

    # Flatten MultiIndex columns
    category_features.columns = ['_'.join(col).strip('_') for col in category_features.columns.values]

    # Rename columns for clarity
    category_features.rename(columns={
        'trace_id_count': 'num_traces',
        'total_response_time_mean': 'avg_total_response_time',
        'total_response_time_max': 'max_total_response_time',
        'max_response_time_mean': 'avg_max_response_time',
        'mean_response_time_mean': 'avg_mean_response_time',
        'total_operations_mean': 'avg_total_operations',
        'total_service_calls_mean': 'avg_total_service_calls',
        'total_ret_mean': 'avg_total_ret',
        'total_ret_max': 'max_total_ret',
        'max_ret_mean': 'avg_max_ret',
        'mean_ret_mean': 'avg_mean_ret',
    }, inplace=True)

    # Display the aggregated features for each trace category
    print("Aggregated Features by Trace Category:")
    print(category_features.head())


def detect_blob_traces(trace_features_df):
    # Thresholds can be adjusted based on data distribution or known Blob behaviors
    high_service_call_threshold = trace_features_df['total_service_calls'].quantile(0.75)  # Top 25% calls
    high_response_time_threshold = trace_features_df['max_response_time'].quantile(0.75)  # Top 25% response times
    high_ret_threshold = trace_features_df['total_ret'].quantile(0.75)  # Top 25% total ret values

    # Detect Blob traces based on high service call frequency, response times, and data size
    blob_traces = trace_features_df[
        (trace_features_df['total_service_calls'] > high_service_call_threshold) &
        (trace_features_df['max_response_time'] > high_response_time_threshold) &
        (trace_features_df['total_ret'] > high_ret_threshold)
        ]

    # Display the detected Blob traces
    print(f"Detected {len(blob_traces)} Blob traces based on high service calls, response times, and data size.")
    print(blob_traces[['trace_id', 'trace_type', 'total_service_calls', 'max_response_time', 'total_ret']])

    return blob_traces


def compute_baselines(trace_features_df):
    # Compute the baseline mean and standard deviation for each trace category
    baselines = trace_features_df.groupby('trace_type').agg({
        'total_service_calls': ['mean', 'std'],
        'max_response_time': ['mean', 'std'],
        'total_ret': ['mean', 'std'],
    }).reset_index()

    # Flatten the column names
    baselines.columns = ['trace_type', 'mean_service_calls', 'std_service_calls',
                         'mean_max_response_time', 'std_max_response_time',
                         'mean_total_ret', 'std_total_ret']

    return baselines


def detect_anomalies(trace_features_df, baselines):
    # Merge trace features with baselines
    merged_df = pd.merge(trace_features_df, baselines, on='trace_type')

    # Calculate Z-scores for anomalies
    merged_df['z_service_calls'] = (merged_df['total_service_calls'] - merged_df['mean_service_calls']) / merged_df[
        'std_service_calls']
    merged_df['z_response_time'] = (merged_df['max_response_time'] - merged_df['mean_max_response_time']) / merged_df[
        'std_max_response_time']
    merged_df['z_total_ret'] = (merged_df['total_ret'] - merged_df['mean_total_ret']) / merged_df['std_total_ret']

    # Flag anomalies (using Z-scores > 3 as a threshold, meaning the value is more than 3 standard deviations away)
    anomalies = merged_df[(merged_df['z_service_calls'] > 3) |
                          (merged_df['z_response_time'] > 3) |
                          (merged_df['z_total_ret'] > 3)]

    print(f"Detected {len(anomalies)} anomalous traces.")
    return anomalies


def analysis(category_features, trace_features_df):
    # Display the aggregated features
    print("\nAggregated Features by Trace Category:")
    print(category_features)

    # Optionally, save the aggregated features to a CSV file for further analysis
    # category_features.to_csv('trace_category_features.csv', index=False)

    # Initialize a dictionary to store pairwise service call counts per category
    category_pairwise_calls = {}

    # Iterate over each category
    for category, group_df in trace_features_df.groupby('trace_type'):
        # Counter to accumulate pairwise calls
        pairwise_counter = Counter()
        for calls in group_df['pairwise_service_calls']:
            # Each 'calls' is a string like 'serviceA->serviceB:count; serviceC->serviceD:count; ...'
            for call in calls.split('; '):
                if call:
                    pair, count = call.split(':')
                    pairwise_counter[pair] += int(count)
        # Store the counter in the dictionary
        category_pairwise_calls[category] = pairwise_counter

    # Display pairwise service call counts for each category
    print("\nPairwise Service Call Counts per Category:")
    for category, counter in category_pairwise_calls.items():
        print(f"\nTrace Type {category}:")
        for pair, count in counter.items():
            print(f"  {pair}: {count}")


from sklearn.ensemble import IsolationForest


def run_isolation_forest(trace_features_df):
    # Select relevant numerical features
    features = trace_features_df[['total_service_calls', 'max_response_time', 'total_ret']]

    # Fit Isolation Forest
    model = IsolationForest(contamination=0.05)  # Set contamination rate (percentage of anomalies)
    trace_features_df['anomaly'] = model.fit_predict(features)

    # Flag traces labeled as anomalies (-1)
    anomalies = trace_features_df[trace_features_df['anomaly'] == -1]

    print(f"Detected {len(anomalies)} anomalous traces using Isolation Forest.")
    return anomalies

def generate_anomaly_report(anomalies):
    report = anomalies[['trace_id', 'trace_type', 'total_service_calls',
                        'max_response_time', 'total_ret',
                        'z_service_calls', 'z_response_time', 'z_total_ret']]
    print("Anomaly Report:")
    print(report)
    return report


def load_data(folder_path):
    dataframes = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            # Load the dataset from the CSV file
            df = pd.read_csv(file_path)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Main function to orchestrate the workflow
def main():
    baseline_folder = sys.argv[1]
    blob_folder = sys.argv[2]
    # est_folder = sys.argv[3]
    # Load data
    baseline_df = load_data(baseline_folder)
    blob_df = load_data(blob_folder)
    # est_df = load_data(est_folder)

    # Visualize the distribution of response times and message sizes
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(baseline_df['response_time'], kde=True, color='blue')
    plt.title('Distribution of Response Times')

    plt.subplot(1, 2, 2)
    sns.histplot(baseline_df['ret'], kde=True, color='green')
    plt.title('Distribution of Message Sizes (ret)')

    plt.tight_layout()
    plt.savefig('BAselineDistribution.png', dpi=300, bbox_inches='tight')

    # Visualize the distribution of response times and message sizes
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(blob_df['response_time'], kde=True, color='blue')
    plt.title('Distribution of Response Times')

    plt.subplot(1, 2, 2)
    sns.histplot(blob_df['ret'], kde=True, color='green')
    plt.title('Distribution of Message Sizes (ret)')

    plt.tight_layout()
    plt.savefig('BlobDistribution.png', dpi=300, bbox_inches='tight')


    baseline_classified= classifiy_traces(baseline_df)
    blob_classified= classifiy_traces(blob_df)


    trace_features_df_baseline= extract_features(baseline_classified)
    trace_features_df_blob= extract_features(blob_classified)

    category_features_baseline= aggregate_features(trace_features_df_baseline)
    category_features_blob= aggregate_features(trace_features_df_blob)

    analysis(category_features_baseline, trace_features_df_baseline)
    analysis(category_features_blob, trace_features_df_blob)

    baselines = compute_baselines(trace_features_df_baseline)

    # Run the anomaly detection
    anomalies = detect_anomalies(trace_features_df_blob, baselines)

    # Run Isolation Forest anomaly detection
    isolation_forest_anomalies = run_isolation_forest(trace_features_df_blob)

    # Generate the anomaly report
    anomaly_report = generate_anomaly_report(anomalies)


if __name__ == "__main__":
    main()
