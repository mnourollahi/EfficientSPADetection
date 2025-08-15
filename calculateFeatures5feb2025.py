import networkx as nx
import sys
import pandas as pd
import os
import multiprocessing as mp
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation
import networkx as nx
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests


def process_service_level_data(df, time_window=100000000000):
    """
    Aggregate service-level statistics per time window.
    """
    df["time_bucket"] = (df["time_stamp"] // time_window) * time_window  # Create time buckets

    df_service = df.groupby(["service_name", "time_bucket"]).agg({
        "trace_id": "nunique",  # Total unique requests per service (proxy for throughput)
        "span_duration": ["mean", "sum"],  # Average and total span duration
        "recv_data": "sum",
        "transmitted_data": "sum",
        "recv_syscall_count": "sum",
        "send_syscall_count": "sum",
        "total_operations": "sum",  # Total operations processed
        "error_count": "sum",  # Total errors encountered
        "service_contribution": "mean"  # Proxy for centrality (Betweenness)
    }).reset_index()

    # Rename columns for clarity
    df_service.columns = [
        "service_name", "time_bucket", "total_requests", "avg_span_duration", "sum_span_duration",
        "total_recv_data", "total_transmitted_data", "total_recv_syscalls", "total_send_syscalls",
        "total_operations", "total_errors", "avg_service_contribution"
    ]

    return df_service


def process_operation_level_data(df):
    # Aggregate operation-level statistics
    df_operation = df.groupby(["service_name", "op_name"]).agg({
        "trace_id": "nunique",  # Total unique requests per operation
        "span_duration": ["mean", "sum"],  # Average and total span duration
        "recv_data": "sum",
        "transmitted_data": "sum",
        "recv_syscall_count": "sum",
        "send_syscall_count": "sum",
        "total_operations": "sum",  # Total operations processed
        "error_count": "sum",  # Total errors encountered
    }).reset_index()

    # Rename columns for clarity
    df_operation.columns = [
        "service_name", "op_name", "total_requests", "avg_span_duration", "sum_span_duration",
        "total_recv_data", "total_transmitted_data", "total_recv_syscalls", "total_send_syscalls",
        "total_operations", "total_errors"
    ]

    return df_operation


def service_dependency_graph(df, df_service):
    # Create a directed graph for service dependencies
    G = nx.DiGraph()

    # Extract pairwise interactions from the dataset
    service_pairs = df.groupby(["service_name", "op_name"])["trace_id"].nunique().reset_index()
    service_pairs.columns = ["source_service", "operation", "interaction_count"]

    # Add edges with weights (interaction frequency)
    for _, row in service_pairs.iterrows():
        G.add_edge(row["source_service"], row["operation"], weight=row["interaction_count"])

    # Compute graph metrics
    betweenness_centrality = nx.betweenness_centrality(G, weight="weight")
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    # Add centrality measures to the service-level dataset
    df_service["betweenness"] = df_service["service_name"].map(betweenness_centrality)
    df_service["degree"] = df_service["service_name"].map(degree_centrality)
    df_service["closeness"] = df_service["service_name"].map(closeness_centrality)

    return df_service

def operation_dependency_graph(df, df_operation):

    return df_operation


# Define percentile-based thresholds for each anti-pattern
def compute_thresholds(df_service, df_operation):
    return {
        # Blob Detection (Service Level)
        "high_betweenness": df_service["betweenness"].quantile(0.75),
        "high_degree": df_service["degree"].quantile(0.75),
        "high_service_contribution": df_service["avg_service_contribution"].quantile(0.75),
        "high_total_requests": df_service["total_requests"].quantile(0.75),

        # Chatty Detection (Operation Level)
        "high_pairwise_requests": df_operation["pairwise_request_count"].quantile(0.75),
        "high_pairwise_requests": df_operation["total_operations"].quantile(0.75),
        "high_avg_response_time": df_operation["avg_span_duration"].quantile(0.75),
        "high_degree_centrality": df_operation["total_operations"].quantile(0.75),

        # Empty Semi-Truck Detection (Operation Level)
        "high_send_syscalls": df_operation["total_send_syscalls"].quantile(0.75),
        "low_transmitted_data_per_call": (
                    df_operation["total_transmitted_data"] / df_operation["total_send_syscalls"]).quantile(0.25)
    }

def compute_pairwise_request_count(df):
    """
    Compute pairwise request counts: how often a service calls another.
    """
    # Ensure required columns exist
    if "trace_id" not in df or "service_name" not in df:
        raise ValueError("Missing required columns: 'trace_id' and 'service_name'")

    # Get service interactions within each trace
    pairwise_interactions = (
        df.groupby(["trace_id", "service_name"])
        .size()
        .reset_index(name="pairwise_request_count")
    )

    # Merge with original dataframe to add the pairwise count per trace
    df = df.merge(pairwise_interactions, on=["trace_id", "service_name"], how="left")

    return df


# Apply updated weak labeling function to dataset
def compute_weak_labels(df_service, df_operation):
    # Compute thresholds dynamically based on percentiles
    thresholds = {
        # Blob Detection (Service Level)
        "high_service_contribution": df_service["avg_service_contribution"].quantile(0.75),
        "high_total_requests": df_service["total_operations"].quantile(0.75),

        # Chatty Detection (Operation Level)
        "high_total_operations": df_operation["total_operations"].quantile(0.75),
        "high_avg_response_time": df_operation["span_duration"].quantile(0.75),

        # Empty Semi-Truck Detection (Operation Level)
        "high_send_syscalls": df_operation["send_syscall_count"].quantile(0.75),
        "low_transmitted_data_per_call": (
            df_operation["transmitted_data"] / df_operation["send_syscall_count"]
        ).quantile(0.25),
    }

    # Apply weak labeling for Blob (Service Level)
    df_service["weak_Blob"] = (
        (df_service["avg_service_contribution"] > thresholds["high_service_contribution"]) |
        (df_service["total_operations"] > thresholds["high_total_requests"])
    )

    # Apply weak labeling for Chatty Services (Operation Level)
    df_operation["weak_Chatty"] = (
        (df_operation["total_operations"] > thresholds["high_total_operations"]) &
        (df_operation["span_duration"] > thresholds["high_avg_response_time"])
    )

    # Apply weak labeling for Empty Semi-Trucks (EST) (Operation Level)
    df_operation["weak_EST"] = (
        (df_operation["send_syscall_count"] > thresholds["high_send_syscalls"]) &
        ((df_operation["transmitted_data"] / df_operation["send_syscall_count"]) < thresholds["low_transmitted_data_per_call"])
    )

    # Assign Weak Labels
    df_service["weak_label"] = "Normal"
    df_service.loc[df_service["weak_Blob"], "weak_label"] = "Blob"

    df_operation["weak_label"] = "Normal"
    df_operation.loc[df_operation["weak_Chatty"], "weak_label"] = "Chatty"
    df_operation.loc[df_operation["weak_EST"], "weak_label"] = "EST"

    return df_service, df_operation


def clustering_blob_detection(df_service):
    """
    Perform clustering for Blob Detection (Service Level).
    """

    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    # New feature selection (since `total_requests` is per-request)
    features_blob = ["betweenness", "degree", "avg_service_contribution", "sum_span_duration"]

    # Ensure features exist in the dataset
    available_features = [f for f in features_blob if f in df_service.columns]

    if not available_features:
        print("Error: None of the selected features are available in df_service!")
        return df_service

    X_blob = df_service[available_features].fillna(0).values

    scaler_blob = StandardScaler()
    X_blob_scaled = scaler_blob.fit_transform(X_blob)

    dbscan_blob = DBSCAN(eps=0.8, min_samples=2)
    clusters_blob = dbscan_blob.fit_predict(X_blob_scaled)

    df_service["cluster_label"] = clusters_blob
    return df_service


from sklearn.cluster import MiniBatchKMeans


def clustering_est_chatty_detection(df_service, df_operation, batch_size=5000):
    """
    Perform clustering for EST & Chatty Detection (Operation Level) using Mini-Batch KMeans.
    """
    features_operation = ["total_send_syscalls", "total_transmitted_data", "total_operations", "avg_span_duration"]

    # Filter only available columns
    available_features = [f for f in features_operation if f in df_operation.columns]
    if not available_features:
        print("Error: None of the selected features are available in df_operation!")
        return df_service, df_operation

    X_operation = df_operation[available_features].fillna(0).values

    # Scale features
    scaler_operation = StandardScaler()
    X_operation_scaled = scaler_operation.fit_transform(X_operation)

    # Use Mini-Batch KMeans instead of DBSCAN
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=batch_size, random_state=42)
    clusters_operation = kmeans.fit_predict(X_operation_scaled)

    df_operation["cluster_label"] = clusters_operation

    return df_service, df_operation


def label_adjustments(df_service, df_operation):
    # Compare Weak Labels with Cluster Assignments

    # Service-Level: Count of weak labels vs. cluster labels
    service_cluster_comparison = df_service.groupby(["cluster_label", "weak_label"]).size().unstack(fill_value=0)

    # Operation-Level: Count of weak labels vs. cluster labels
    operation_cluster_comparison = df_operation.groupby(["cluster_label", "weak_label"]).size().unstack(fill_value=0)

    # Apply majority voting to refine labels
    df_service = adjust_labels(df_service, service_cluster_comparison)
    df_operation = adjust_labels(df_operation, operation_cluster_comparison)
    return df_service, df_operation

# Apply Majority Voting for Label Adjustment

def adjust_labels(df, cluster_comparison):
    """
    Adjust labels based on majority voting within clusters.
    """
    updated_labels = df["weak_label"].copy()

    # Iterate through clusters and adjust labels based on majority class
    for cluster in cluster_comparison.index:
        majority_label = cluster_comparison.loc[cluster].idxmax()  # Find most common label in cluster
        df.loc[df["cluster_label"] == cluster, "refined_label"] = majority_label

    return df

def semi_supervised_learning(df_service, df_operation):
    # Convert weak labels to numeric for Label Propagation
    label_mapping = {"Normal": 0, "Blob": 1, "EST": 2, "Chatty": 3}

    df_operation["semi_supervised_label"] = df_operation["refined_label"].map(label_mapping)

    # Replace unclassified entries (-1 cluster) with -1 label for semi-supervised learning
    df_operation.loc[df_operation["cluster_label"] == -1, "semi_supervised_label"] = -1

    # Select feature columns
    features_operation = ["total_send_syscalls", "total_transmitted_data", "total_operations", "avg_span_duration"]
    X_operation = df_operation[features_operation].fillna(0).values

    # Standardize features
    scaler_operation = StandardScaler()
    X_operation_scaled = scaler_operation.fit_transform(X_operation)

    # Train Label Propagation model
    label_prop_model = LabelPropagation()
    label_prop_model.fit(X_operation_scaled, df_operation["semi_supervised_label"])

    # Predict refined labels
    df_operation["final_label"] = label_prop_model.predict(X_operation_scaled)

    # Map labels back to readable format
    reverse_label_mapping = {0: "Normal", 1: "Blob", 2: "EST", 3: "Chatty"}
    df_operation["final_label"] = df_operation["final_label"].map(reverse_label_mapping)

    # Count of final labels
    df_operation["final_label"].value_counts()
    return df_service, df_operation

# def construct_causal_graph(df_service):
#     """
#     Construct a weighted causal graph using per-time-window data.
#     """
#     G_causal = nx.DiGraph()
#     causality_scores = {}
#
#     # Iterate through each time bucket
#     for time_bucket in df_service["time_bucket"].unique():
#         df_time = df_service[df_service["time_bucket"] == time_bucket]
#
#         # Compute causality for each time slice
#         for source in df_time["service_name"]:
#             for target in df_time["service_name"]:
#                 if source != target:
#                     try:
#                         source_series = df_time[df_time["service_name"] == source]["sum_span_duration"].values
#                         target_series = df_time[df_time["service_name"] == target]["sum_span_duration"].values
#
#                         if len(source_series) > 1 and len(target_series) > 1:
#                             test_result = grangercausalitytests(
#                                 np.column_stack([source_series, target_series]), maxlag=1, verbose=False
#                             )
#                             p_value = test_result[1][0]["ssr_ftest"][1]
#                             causality_strength = 1 - p_value if p_value < 0.05 else 0
#
#                             causality_scores[(source, target, time_bucket)] = causality_strength
#                             G_causal.add_edge(source, target, weight=causality_strength)
#                     except:
#                         continue
#
#     # Normalize betweenness centrality
#     df_service = normalize_centrality(G_causal, df_service)
#
#     # Compute Root Cause Score per time bucket
#     df_service["root_cause_score"] = df_service.apply(
#         lambda row: sum(
#             causality_scores.get((row["service_name"], target, row["time_bucket"]), 0)
#             for target in df_service["service_name"]
#         ) + row["betweenness"] * 0.2 + row["total_errors"] * 0.1,
#         axis=1,
#     )
#
#     return df_service.sort_values(by=["time_bucket", "root_cause_score"], ascending=[True, False])
#

import multiprocessing as mp
from functools import partial


def granger_causality_worker(pair, df_service):
    """
    Function for parallel execution of Granger Causality.
    """
    source, target = pair
    try:
        # Extract time series data for source and target
        source_series = df_service[df_service["service_name"] == source]["sum_span_duration"].values
        target_series = df_service[df_service["service_name"] == target]["sum_span_duration"].values

        # Skip if there are too few values for Granger test
        if len(source_series) <= 3 or len(target_series) <= 3:
            return (source, target, 0)

        # Perform Granger causality test
        test_result = grangercausalitytests(np.column_stack([source_series, target_series]), maxlag=1, verbose=False)
        p_value = test_result[1][0]["ssr_ftest"][1]  # Extract p-value

        causality_strength = 1 - p_value if p_value < 0.05 else 0
        return (source, target, causality_strength)

    except Exception as e:
        return (source, target, 0)  # If an error occurs, return zero causality


def compute_causality_scores(df_service, num_workers=4, min_interactions=5):
    """
    Computes causality scores using multiprocessing for better performance.

    Args:
        df_service (pd.DataFrame): Aggregated service-level data.
        num_workers (int): Number of parallel processes.
        min_interactions (int): Minimum number of interactions before testing.

    Returns:
        dict: Causality scores dictionary {(source, target): causality_score}.
    """
    causality_scores = {}

    # Step 1: Identify Service Pairs with Sufficient Data
    interaction_counts = df_service.groupby("service_name")["total_requests"].sum()
    high_interaction_services = interaction_counts[interaction_counts > min_interactions].index.tolist()

    # Generate service pairs for Granger causality test
    service_pairs = [(s1, s2) for s1 in high_interaction_services for s2 in high_interaction_services if s1 != s2]

    print(f"Processing {len(service_pairs)} service pairs with multiprocessing...")

    # Step 2: Use Multiprocessing for Speedup
    with mp.Pool(num_workers) as pool:
        results = pool.map(partial(granger_causality_worker, df_service=df_service), service_pairs)

    # Convert results into dictionary
    for source, target, score in results:
        causality_scores[(source, target)] = score

    return causality_scores


def normalize_centrality(G, df_service):
    if len(G.nodes) == 0:
        print("Warning: Causal Graph is empty. Setting betweenness centrality to 0.")
        df_service["betweenness"] = 0  # Assign 0 if graph is empty
        return df_service

    betweenness = nx.betweenness_centrality(G, weight="weight")

    # Check if betweenness is empty
    if not betweenness:
        df_service["betweenness"] = 0  # Assign 0 if no centrality values exist
        return df_service

    # Normalize betweenness centrality
    max_betweenness = max(betweenness.values()) if max(betweenness.values()) > 0 else 1
    df_service["betweenness"] = df_service["service_name"].map(lambda s: betweenness.get(s, 0) / max_betweenness)

    return df_service

import networkx as nx
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

def construct_causal_graph(df_service):
    """
    Construct a weighted causal graph based on service-level time-bucketed data.
    Removes `trace_id` dependence and uses `time_bucket` for causality calculations.
    """

    G_causal = nx.DiGraph()
    causality_scores = {}

    if "time_bucket" not in df_service.columns:
        raise ValueError("Missing 'time_bucket' column in df_service. Ensure time-based aggregation is used.")

    for time_bucket in df_service["time_bucket"].unique():
        df_time = df_service[df_service["time_bucket"] == time_bucket]

        print(f"Processing Time Bucket: {time_bucket} - {len(df_time)} services found")

        if len(df_time) < 2:
            print(f"Skipping time bucket {time_bucket} due to insufficient services.")
            continue  # Skip if there are not enough services to establish causality

        for source in df_time["service_name"]:
            for target in df_time["service_name"]:
                if source != target:
                    try:
                        source_series = df_time[df_time["service_name"] == source]["sum_span_duration"].values
                        target_series = df_time[df_time["service_name"] == target]["sum_span_duration"].values

                        print(f"Checking causality between {source} -> {target} in time bucket {time_bucket}")

                        if len(source_series) > 1 and len(target_series) > 1:
                            test_result = grangercausalitytests(
                                np.column_stack([source_series, target_series]), maxlag=1, verbose=False
                            )
                            p_value = test_result[1][0]["ssr_ftest"][1]
                            causality_strength = 1 - p_value if p_value < 0.05 else 0

                            if causality_strength > 0:
                                print(f"Causal edge added: {source} -> {target} with weight {causality_strength}")
                                causality_scores[(source, target, time_bucket)] = causality_strength
                                G_causal.add_edge(source, target, weight=causality_strength)
                            else:
                                print(f"No significant causality detected for {source} -> {target}")

                    except Exception as e:
                        print(f"Error computing causality for {source} -> {target}: {e}")
                        continue

    # Check if graph has any edges
    if len(G_causal.edges) == 0:
        print("Warning: Causal Graph is empty. Setting betweenness centrality to 0.")
        df_service["betweenness"] = 0
    else:
        df_service = normalize_centrality(G_causal, df_service)

    df_service["root_cause_score"] = df_service.apply(
        lambda row: sum(
            causality_scores.get((row["service_name"], target, row["time_bucket"]), 0)
            for target in df_service["service_name"]
        ) + row["betweenness"] * 0.2 + row["total_errors"] * 0.1,
        axis=1,
    )

    return df_service.sort_values(by=["time_bucket", "root_cause_score"], ascending=[True, False])


def process_all_csv_in_folder(folder_path):
    # List all .txt files in the folder
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".txt")]

    # Read all files into a list of DataFrames
    dataframes = [pd.read_csv(file) for file in csv_files]

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df


def optimize_memory(df):
    """
    Convert large numeric columns to smaller data types.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")

    return df

if __name__ == "__main__":
    #process_trace_data('input.csv', 'output.csv')
    input_folder = sys.argv[1]
    # output_folder = sys.argv[2]
    df= process_all_csv_in_folder(input_folder)
    df = optimize_memory(df)
    df_service= process_service_level_data(df, 1000000)
    df_operation= process_operation_level_data(df)
    df_operation = process_operation_level_data(df)
    df_operation = compute_pairwise_request_count(df)
    df_service = service_dependency_graph(df, df_service)
    df_service, df_operation= compute_weak_labels(df_service, df_operation)
    print("weak labeling")

    # Return weak label counts
    weak_label_counts_service = df_service["weak_label"].value_counts()
    weak_label_counts_operation = df_operation["weak_label"].value_counts()

    print(weak_label_counts_service)
    print(weak_label_counts_operation)
    df_service.to_csv("df_service1", index=False)
    # df_operation.to_csv("2", index=False)
    df_service= clustering_blob_detection(df_service)
    print("clustering blob labeling")
    df_service, df_operation= clustering_est_chatty_detection(df_service, df_operation, batch_size=5000)
    print("clusterting hte rest  labeling")
    df_service, df_operation= label_adjustments(df_service, df_operation)
    print("label adjustments...")
    # df_service, df_operation= semi_supervised_learning(df_service, df_operation)
    df_service["error_contribution"] = df_service["total_errors"] / (df_service["total_errors"].sum() + 1)

    causality_scores = compute_causality_scores(df_service, num_workers=8, min_interactions=10)
    # Compute causal graph without using trace_id
    print("Computing causality graph based on time-bucketed service data...")
    df_cause_sorted = construct_causal_graph(df_service)
    print("Causal graph constructed!")
    df_service.to_csv("df_cause_sorted", index=False)
    # df_operation.to_csv("df_operation", index=False)
