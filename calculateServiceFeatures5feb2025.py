import networkx as nx
import sys
import pandas as pd
import os
import multiprocessing as mp
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation


def process_service_level_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Aggregate service-level statistics
    df_service = df.groupby("service_name").agg({
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
        "service_name", "total_requests", "avg_span_duration", "sum_span_duration",
        "total_recv_data", "total_transmitted_data", "total_recv_syscalls", "total_send_syscalls",
        "total_operations", "total_errors", "avg_service_contribution"
    ]

    # Compute thresholds
    service_thresholds = compute_service_level_thresholds(df_service)

    # Apply weak labeling rules
    df_service["weak_EST"] = (
            (df_service["total_send_syscalls"] > service_thresholds["high_send_syscalls"]) &
            (df_service["total_transmitted_data"] < service_thresholds["low_transmitted_data"])
    )

    df_service["weak_Chatty"] = (
            (df_service["total_operations"] > service_thresholds["high_pairwise_operations"]) &
            (df_service["avg_span_duration"] > service_thresholds["high_span_duration"])
    )

    df_service["weak_Blob"] = (
        (df_service["avg_service_contribution"] > service_thresholds["high_betweenness"])
    )

    # Assign weak labels based on conditions
    df_service["weak_label"] = "Normal"
    df_service.loc[df_service["weak_EST"], "weak_label"] = "EST"
    df_service.loc[df_service["weak_Chatty"], "weak_label"] = "Chatty"
    df_service.loc[df_service["weak_Blob"], "weak_label"] = "Blob"

    # Display counts of each weak label
    weak_label_counts_service = df_service["weak_label"].value_counts()
    # Count of weak labels
    df_service["weak_label"].value_counts()

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

    # Compute thresholds
    operation_thresholds = compute_operation_level_thresholds(df_operation)

    # Apply weak labeling rules
    df_operation["weak_EST"] = (
            (df_operation["total_send_syscalls"] > operation_thresholds["high_send_syscalls"]) &
            (df_operation["total_transmitted_data"] < operation_thresholds["low_transmitted_data"])
    )

    df_operation["weak_Chatty"] = (
            (df_operation["total_operations"] > operation_thresholds["high_pairwise_operations"]) &
            (df_operation["avg_span_duration"] > operation_thresholds["high_span_duration"])
    )

    # Assign weak labels based on conditions
    df_operation["weak_label"] = "Normal"
    df_operation.loc[df_operation["weak_EST"], "weak_label"] = "EST"
    df_operation.loc[df_operation["weak_Chatty"], "weak_label"] = "Chatty"

    # Count of weak labels
    df_operation["weak_label"].value_counts()

    return df_operation


# Define adaptive thresholds using percentiles for weak labeling at the operation level
def compute_operation_level_thresholds(df_operation):
    return {
        "high_send_syscalls": df_operation["total_send_syscalls"].quantile(0.75),  # High outgoing messages
        "low_transmitted_data": df_operation["total_transmitted_data"].quantile(0.25),  # Low data sent
        "high_pairwise_operations": df_operation["total_operations"].quantile(0.75),  # High operation count
        "high_span_duration": df_operation["avg_span_duration"].quantile(0.75)  # High execution time
    }

# Define adaptive thresholds using percentiles for weak labeling
def compute_service_level_thresholds(df_service):
    return {
        "high_send_syscalls": df_service["total_send_syscalls"].quantile(0.75),  # High number of outgoing messages
        "low_transmitted_data": df_service["total_transmitted_data"].quantile(0.25),  # Low total data sent
        "high_pairwise_operations": df_service["total_operations"].quantile(0.75),  # High operation count
        "high_span_duration": df_service["avg_span_duration"].quantile(0.75),  # High execution time
        "high_betweenness": df_service["avg_service_contribution"].quantile(0.75)  # High service centrality (Blob)
    }


def service_dependency_graph(df, df_service):

    # Create a directed graph of service dependencies (Service A -> Service B interactions)
    G = nx.DiGraph()

    # Extract pairwise interactions (assume that services interact based on trace relationships)
    service_pairs = df[["service_name", "op_name"]].groupby("service_name").count().reset_index()
    service_pairs.columns = ["source_service", "interaction_count"]

    # Add edges with weights (interaction frequency)
    for _, row in service_pairs.iterrows():
        G.add_edge(row["source_service"], row["source_service"], weight=row["interaction_count"])

    # Compute centrality metrics
    betweenness_centrality = nx.betweenness_centrality(G, weight="weight")
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    # Add centrality measures to the service-level dataset
    df_service["betweenness"] = df_service["service_name"].map(betweenness_centrality)
    df_service["degree"] = df_service["service_name"].map(degree_centrality)
    df_service["closeness"] = df_service["service_name"].map(closeness_centrality)
    return df_service

def blob_labeling(df_service):
    # Compute thresholds
    blob_thresholds = compute_blob_thresholds(df_service)

    # Apply weak labeling for Blob (using graph metrics)
    df_service["weak_Blob"] = (
            (df_service["betweenness"] > blob_thresholds["high_betweenness"]) |
            (df_service["degree"] > blob_thresholds["high_degree"]) |
            (df_service["avg_service_contribution"] > blob_thresholds["high_service_contribution"])
    )

    # Assign updated weak labels for Blob
    df_service["weak_label"] = df_service["weak_label"].where(~df_service["weak_Blob"], "Blob")
    # Count of refined Blob labels
    df_service["weak_label"].value_counts()

    return df_service

def operation_dependency_graph(df, df_operation):
    # Compute operation-level degree centrality for Chatty detection
    G_op = nx.DiGraph()

    # Extract pairwise interactions from the dataset
    operation_pairs = df.groupby(["service_name", "op_name"])["trace_id"].nunique().reset_index()
    operation_pairs.columns = ["source_service", "operation", "interaction_count"]

    # Add edges with weights (interaction frequency)
    for _, row in operation_pairs.iterrows():
        G_op.add_edge(row["source_service"], row["operation"], weight=row["interaction_count"])

    # Compute operation-level degree centrality
    operation_degree_centrality = nx.degree_centrality(G_op)

    # Map centrality to operation-level dataset
    df_operation["degree_centrality"] = df_operation["op_name"].map(operation_degree_centrality)
    return df_operation

def compute_blob_thresholds(df_service):
    return {
        "high_betweenness": df_service["betweenness"].quantile(0.75),
        "high_degree": df_service["degree"].quantile(0.75),
        "high_service_contribution": df_service["avg_service_contribution"].quantile(0.75),
    }


def clustering_validate_weak_label(df_service, df_operaton):
    # Select relevant features for clustering
    features_service = ["total_send_syscalls", "total_transmitted_data", "total_operations",
                        "avg_span_duration", "avg_service_contribution"]
    X_service = df_service[features_service].fillna(0).values  # Ensure no NaN values

    # Standardize the features for better clustering performance
    scaler_service = StandardScaler()
    X_scaled_service = scaler_service.fit_transform(X_service)

    # Apply DBSCAN clustering (adjust hyperparameters as needed)
    dbscan_service = DBSCAN(eps=0.8, min_samples=2)  # eps and min_samples can be tuned
    clusters_service = dbscan_service.fit_predict(X_scaled_service)

    # Add cluster labels to the service-level dataset
    df_service["cluster_label"] = clusters_service

    # Compare weak labels with cluster assignments
    cluster_label_counts_service = df_service["cluster_label"].value_counts()

    return df_service

def semi_supervised(df_service):

    # Prepare labeled and unlabeled data for semi-supervised learning
    df_service["semi_supervised_label"] = df_service["weak_label"].map({"Normal": 0, "Blob": 1, "Chatty": 2, "EST": 3})

    # Convert weak labels to numeric, keeping -1 for unlabeled data
    df_service.loc[df_service["cluster_label"] == -1, "semi_supervised_label"] = -1  # Treat outliers as unlabeled

    # Prepare feature matrix (same as clustering step)
    X_train_service = scaler_service.transform(df_service[features_service])  # Ensure same scaling

    # Train Label Propagation model
    label_prop_model = LabelPropagation()
    label_prop_model.fit(X_train_service, df_service["semi_supervised_label"])

    # Predict new labels
    df_service["refined_label"] = label_prop_model.predict(X_train_service)

    # Map labels back to readable form
    label_mapping = {0: "Normal", 1: "Blob", 2: "Chatty", 3: "EST"}
    df_service["refined_label"] = df_service["refined_label"].map(label_mapping)

    # Display the refined labels
    tools.display_dataframe_to_user(name="Refined Service-Level Labels", dataframe=df_service)

    # Count of new refined labels
    df_service["refined_label"].value_counts()


def process_csv_file(args):
    input_csv, output_csv = args
    df_service = process_service_level_data(input_csv, output_csv)
    df_service_clustered= clustering_validate_weak_label(df_service)


def process_all_csv_in_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = [(os.path.join(folder_path, file), os.path.join(output_folder, f"processed_{file}"))
                 for file in os.listdir(folder_path) if file.endswith(".txt")]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_csv_file, csv_files)


if __name__ == "__main__":
    #process_trace_data('input.csv', 'output.csv')
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    process_all_csv_in_folder(input_folder, output_folder)
