
import os
import matplotlib.pyplot as plt
import sys
import pandas as pd
import networkx as nx
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns


def compute_weak_labels(df_request):
    """
    Compute weak labels per request without aggregation.
    """

    # Compute dynamic thresholds based on per-request data
    thresholds = {
        # Blob Detection (Per Request)
        "high_service_contribution": df_request["service_contribution"].quantile(0.75),
        "high_total_operations": df_request["total_operations"].quantile(0.75),

        # Chatty Services (Per Request)
        "high_total_operations": df_request["total_operations"].quantile(0.75),
        "high_avg_response_time": df_request["span_duration"].quantile(0.75),

        # Empty Semi-Truck (EST) (Per Request)
        "high_send_syscalls": df_request["send_syscall_count"].quantile(0.75),
        "low_transmitted_data_per_call": (
            df_request["transmitted_data"] / df_request["send_syscall_count"]
        ).quantile(0.25),
    }

    # 🚨 Apply weak labeling **per request** instead of aggregating
    df_request["weak_Blob"] = (
        (df_request["service_contribution"] > thresholds["high_service_contribution"]) |
        (df_request["total_operations"] > thresholds["high_total_operations"])
    )

    df_request["weak_Chatty"] = (
        (df_request["total_operations"] > thresholds["high_total_operations"]) &
        (df_request["span_duration"] > thresholds["high_avg_response_time"])
    )

    df_request["weak_EST"] = (
        (df_request["send_syscall_count"] > thresholds["high_send_syscalls"]) &
        ((df_request["transmitted_data"] / df_request["send_syscall_count"]) < thresholds["low_transmitted_data_per_call"])
    )

    # Assign Weak Labels
    df_request["weak_label"] = "Normal"
    df_request.loc[df_request["weak_Blob"], "weak_label"] = "Blob"
    df_request.loc[df_request["weak_Chatty"], "weak_label"] = "Chatty"
    df_request.loc[df_request["weak_EST"], "weak_label"] = "EST"

    print("\n🔹 **Weak Label Distribution:**")
    print(df_request["weak_label"].value_counts())

    return df_request  # ✅ The output remains df_request


from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation


def refine_weak_labels(df_request):
    """
    Refine weak labels using semi-supervised learning (Label Propagation).
    """

    # Step 1: Convert weak labels to numerical values
    label_mapping = {"Normal": 0, "Blob": 1, "EST": 2, "Chatty": 3}
    df_request["weak_label_num"] = df_request["weak_label"].map(label_mapping)

    # Step 2: Assign -1 to uncertain cases (to be inferred)
    df_request.loc[df_request["weak_label"] == "Normal", "weak_label_num"] = -1

    # Step 3: Define Features for Refinement (All Key Features)
    features = [
        "span_duration", "recv_data", "transmitted_data", "recv_syscall_count",
        "send_syscall_count", "total_operations", "avg_recv_msg_size", "avg_sent_msg_size",
        "ratio_recv_time", "ratio_sent_time", "service_contribution", "error_count"
    ]

    # Step 4: Ensure features exist
    available_features = [f for f in features if f in df_request.columns]

    if not available_features:
        print("Error: No valid features found in dataset for weak label refinement!")
        return df_request

    X = df_request[available_features].fillna(0).values

    # Step 5: Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 6: Train Label Propagation Model
    label_prop_model = LabelPropagation()
    label_prop_model.fit(X_scaled, df_request["weak_label_num"])

    # Step 7: Predict Refined Labels
    df_request["refined_label_num"] = label_prop_model.predict(X_scaled)

    # Step 8: Convert Back to Label Names
    reverse_mapping = {0: "Normal", 1: "Blob", 2: "EST", 3: "Chatty"}
    df_request["refined_label"] = df_request["refined_label_num"].map(reverse_mapping)

    # ✅ Print Refinement Results
    print("\n🔹 **Refined Label Distribution:**")
    print(df_request["refined_label"].value_counts())

    return df_request



def compute_per_trace_durations(df):
    """
    Recompute sum_span_duration at the per-trace level instead of aggregated service level.
    """
    df_trace_duration = df.groupby("trace_id")["span_duration"].sum().reset_index()
    df_trace_duration.rename(columns={"span_duration": "sum_span_duration"}, inplace=True)

    # Merge back to df
    df = df.merge(df_trace_duration, on="trace_id", how="left")
    return df

def filter_spa_requests(df_service, df_operation):
    """
    Filter requests that exhibit SPA (Service Performance Anti-patterns).
    Only keep requests labeled as Blob, EST, or Chatty.
    """
    # Keep only SPA-related requests
    df_service_filtered = df_service[df_service["weak_label"].isin(["Blob", "EST", "Chatty"])]
    df_operation_filtered = df_operation[df_operation["weak_label"].isin(["Blob", "EST", "Chatty"])]

    print(f"Filtered Service Requests: {len(df_service_filtered)} / {len(df_service)}")
    print(f"Filtered Operation Requests: {len(df_operation_filtered)} / {len(df_operation)}")

    return df_service_filtered, df_operation_filtered


import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests


def create_dynamic_time_buckets(df, min_requests=100):
    """
    Create dynamic time buckets where each bucket contains at least min_requests.
    If the count is lower, expand the time window dynamically.
    """
    df = df.sort_values("time_stamp").reset_index(drop=True)  # Ensure sorted order
    time_buckets = []

    start_idx = 0
    current_idx = 0
    bucket_id = 0

    while current_idx < len(df):
        # Define the current bucket time range
        start_time = df.loc[start_idx, "time_stamp"]
        current_time = df.loc[current_idx, "time_stamp"]

        # Ensure at least `min_requests` in the bucket
        while current_idx < len(df) and (current_idx - start_idx) < min_requests:
            current_idx += 1
            if current_idx < len(df):
                current_time = df.loc[current_idx, "time_stamp"]

        # Assign time bucket ID
        time_buckets.extend([bucket_id] * (current_idx - start_idx))

        # Move to next bucket
        start_idx = current_idx
        bucket_id += 1

    # Ensure the time buckets match the original dataframe length
    if len(time_buckets) != len(df):
        raise ValueError(f"Time bucket assignment mismatch: {len(time_buckets)} vs {len(df)}")

    df["time_bucket"] = time_buckets
    return df


def compute_granger_causality(df_service, alpha=0.7):
    """
    Compute Granger causality for service interactions based on dynamically adjusted time windows.
    """
    causality_scores = {}

    # Create dynamic time windows (ensures each bucket has at least 100 requests)
    df_service = create_dynamic_time_buckets(df_service, min_requests=100)
    print(f"Time bucketing complete. Assigned {df_service['time_bucket'].nunique()} buckets.")

    # Aggregate per dynamic time bucket
    df_time_series = df_service.groupby(["service_name", "time_bucket"]).agg({
        "sum_span_duration": "sum",
        "error_count": "sum"
    }).reset_index()

    # Get unique services
    all_services = df_time_series["service_name"].unique()

    for source in all_services:
        for target in all_services:
            if source != target:
                try:
                    # Extract per-time-window duration & errors
                    source_series = df_time_series[df_time_series["service_name"] == source][
                        ["time_bucket", "sum_span_duration"]]
                    target_series = df_time_series[df_time_series["service_name"] == target][
                        ["time_bucket", "sum_span_duration"]]

                    # Merge on time_bucket to ensure alignment
                    merged = pd.merge(source_series, target_series, on="time_bucket", how="inner",
                                      suffixes=("_source", "_target"))

                    source_values = merged["sum_span_duration_source"].values
                    target_values = merged["sum_span_duration_target"].values

                    # Debugging: Print sample causality test
                    print(f"Checking causality: {source} -> {target}")

                    # Ensure enough data points for time series analysis
                    if len(source_values) > 10 and len(target_values) > 10:
                        test_result = grangercausalitytests(
                            np.column_stack([source_values, target_values]), maxlag=2, verbose=False
                        )
                        p_value = test_result[2][0]["ssr_ftest"][1]  # Use lag 2 for stronger dependency
                        causality_strength = 1 - p_value if p_value < 0.05 else 0

                        causality_scores[(source, target)] = causality_strength
                except Exception as e:
                    print(f"Failed causality test for {source} -> {target}: {e}")
                    causality_scores[(source, target)] = 0

    return causality_scores

def compute_trace_level_centrality(df_service):
    """
    Compute betweenness, degree, and closeness centrality per trace.
    """
    G = nx.DiGraph()

    # Extract service interactions per trace
    service_pairs = df_service.groupby(["trace_id", "service_name"])["total_operations"].sum().reset_index()

    for _, row in service_pairs.iterrows():
        G.add_edge(row["trace_id"], row["service_name"], weight=row["total_operations"])

    # Compute per-trace centrality
    betweenness = nx.betweenness_centrality(G, weight="weight")
    degree = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G)

    # Assign metrics to dataframe
    df_service["betweenness"] = df_service["trace_id"].map(betweenness)
    df_service["degree"] = df_service["trace_id"].map(degree)
    df_service["closeness"] = df_service["trace_id"].map(closeness)

    return df_service

def construct_causal_graph(df_service, causality_scores, alpha=0.7, beta=0.3):
    """
    Construct a weighted causal graph based on causality + graph metrics.
    """
    G_causal = nx.DiGraph()
    root_cause_scores = {}

    for (source, target), causal_strength in causality_scores.items():
        if causal_strength > 0:  # Only add significant causal relationships
            betweenness_source = df_service[df_service["service_name"] == source]["betweenness"].mean()

            edge_weight = (alpha * causal_strength) + (beta * betweenness_source)
            G_causal.add_edge(source, target, weight=edge_weight)

    # Compute PageRank to rank root causes
    pagerank_scores = nx.pagerank(G_causal, alpha=0.85, weight="weight")

    for service, rank in pagerank_scores.items():
        root_cause_scores[service] = rank

    # Assign scores to the dataframe
    df_service["root_cause_score"] = df_service["service_name"].map(root_cause_scores)
    visualize_causal_graph(G_causal)

    return df_service.sort_values(by="root_cause_score", ascending=False)


def visualize_causal_graph(G):
    """
    Draws a directed causal graph with edge weights.
    """
    pos = nx.spring_layout(G)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    plt.title("Causal Graph")
    plt.savefig('Fausal_Graph.png', dpi=300)


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

# Modified plotting functions to match the features available in your existing pipeline

def plot_input_features(df):
    plt.figure(figsize=(15, 5))

    # Plot Message Length (assumed 'transmitted_data')
    plt.subplot(1, 2, 1)
    if "trace_type" in df.columns and "transmitted_data" in df.columns:
        sns.boxplot(x='trace_type', y='transmitted_data', data=df)
        plt.title('Transmitted Data by Trace Type')

    # Plot Total Operations
    plt.subplot(1, 2, 2)
    if "trace_type" in df.columns and "total_operations" in df.columns:
        sns.boxplot(x='trace_type', y='total_operations', data=df)
        plt.title('Total Operations by Trace Type')

    plt.tight_layout()
    plt.savefig('input_features_boxplots_updated.png', dpi=300)
    plt.show()


def plot_service_call_network(df):
    G = nx.DiGraph()
    for trace_id, trace_df in df.groupby('trace_id'):
        services = trace_df['service_name'].tolist()
        for i in range(len(services) - 1):
            from_service = services[i]
            to_service = services[i + 1]
            if from_service != to_service:
                if G.has_edge(from_service, to_service):
                    G[from_service][to_service]['weight'] += 1
                else:
                    G.add_edge(from_service, to_service, weight=1)

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10,
            font_weight='bold', edge_color='gray')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title('Service Call Diagram for Entire Dataset')
    plt.savefig('service_call_network_updated.png', dpi=300)
    plt.show()


def plot_pairwise_service_call_heatmap(df):
    services = df['service_name'].unique()
    pairwise_matrix = pd.DataFrame(0, index=services, columns=services)
    for trace_id, trace_df in df.groupby('trace_id'):
        services_in_trace = trace_df['service_name'].tolist()
        for i in range(len(services_in_trace) - 1):
            from_service = services_in_trace[i]
            to_service = services_in_trace[i + 1]
            if from_service != to_service:
                pairwise_matrix.loc[from_service, to_service] += 1

    plt.figure(figsize=(12, 10))
    sns.heatmap(pairwise_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5, annot_kws={"size": 10})
    plt.title('Pairwise Service Call Heatmap')
    plt.xlabel('To Service')
    plt.ylabel('From Service')
    plt.tight_layout()
    plt.savefig('pairwise_service_call_heatmap_updated.png', dpi=300)

# This is ready to be plugged into your main script
# Example usage: plot_input_features(df_test) after prediction stage




if __name__ == "__main__":
    input_folder = sys.argv[1]
    df = process_all_csv_in_folder(input_folder)
    df = optimize_memory(df)

    # **Step 0: Compute Weak Labels Per Request**
    df_request = compute_weak_labels(df)
    # df_request_refined= refine_weak_labels(df_request)
    print(df_request.groupby("service_name")["span_duration"].nunique())

    # **Step 1: Filter SPA Requests**


    # **Step 2: Merge back with Original Data**
    df_spa = df.merge(df_request[["trace_id"]], on="trace_id", how="inner")

    # **Step 3: Restore Per-Trace Span Duration**
    df_spa = compute_per_trace_durations(df_spa)  # FIX HERE ✅

    # **Step 4: Compute Granger Causality**
    causality_scores = compute_granger_causality(df_spa)
    print("Granger Causality Scores Sample:", list(causality_scores.items())[:5])

    # **Step 5: Compute Trace-Level Centrality**
    df_spa = compute_trace_level_centrality(df_spa)
    print(df_spa[["betweenness", "degree", "closeness"]].describe())

    # **Step 6: Construct Root Cause Graph**
    df_root_cause = construct_causal_graph(df_spa, causality_scores)

    df_root_cause.to_csv("df_root_cause.csv", index=False)
    df_root_cause_filtered = df_root_cause[df_root_cause["weak_label"].isin(["Blob", "EST", "Chatty"])]

    df_root_cause_unique = df_root_cause.drop_duplicates(subset=["service_name", "op_name"], keep="first")
    df_root_cause_unique.to_csv("df_root_cause_unique.csv", index=False)
    df_root_cause_filtered = df_root_cause_unique[df_root_cause_unique["weak_label"].isin(["Blob", "EST", "Chatty"])]
    df_root_cause_filtered.to_csv("df_root_cause_filtered.csv", index=False)

    plot_input_features(df_root_cause)
    plot_service_call_network(df_root_cause)
    plot_pairwise_service_call_heatmap(df_root_cause)

