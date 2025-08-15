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

def process_request_level_data(df):
    """
    Computes features at the per-request level.
    """
    df_request = df.groupby(["trace_id", "service_name", "op_name"]).agg({
        "span_duration": "sum",
        "recv_data": "sum",
        "transmitted_data": "sum",
        "recv_syscall_count": "sum",
        "send_syscall_count": "sum",
        "total_operations": "sum",
        "error_count": "sum"
    }).reset_index()

    return df_request

def compute_weak_labels(df_request):
    """
    Computes weak labels at the per-request level before aggregation.
    """
    thresholds = {
        "high_total_operations": df_request["total_operations"].quantile(0.75),
        "high_avg_response_time": df_request["span_duration"].quantile(0.75),
        "high_send_syscalls": df_request["send_syscall_count"].quantile(0.75),
        "low_transmitted_data_per_call": (
            df_request["transmitted_data"] / df_request["send_syscall_count"]
        ).quantile(0.25)
    }

    df_request["weak_Blob"] = df_request["total_operations"] > thresholds["high_total_operations"]
    df_request["weak_Chatty"] = df_request["span_duration"] > thresholds["high_avg_response_time"]
    df_request["weak_EST"] = (
        (df_request["send_syscall_count"] > thresholds["high_send_syscalls"]) &
        ((df_request["transmitted_data"] / df_request["send_syscall_count"]) < thresholds["low_transmitted_data_per_call"])
    )

    # Ensure the weak_label column exists
    df_request["weak_label"] = "Normal"
    df_request.loc[df_request["weak_Blob"], "weak_label"] = "Blob"
    df_request.loc[df_request["weak_Chatty"], "weak_label"] = "Chatty"
    df_request.loc[df_request["weak_EST"], "weak_label"] = "EST"

    print("Weak Label Counts:\n", df_request["weak_label"].value_counts())

    return df_request

from sklearn.ensemble import RandomForestClassifier

def train_classification_model(df_request):
    """
    Train a classifier on the weak labels.
    """
    features = ["span_duration", "recv_data", "transmitted_data", "recv_syscall_count", "send_syscall_count", "total_operations"]
    X = df_request[features]
    y = df_request["weak_label"].map({"Normal": 0, "Blob": 1, "EST": 2, "Chatty": 3})  # Convert to numerical labels

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

def load_model_and_predict(model, df_test):
    # Define the feature columns used during training
    feature_columns = ["betweenness", "degree", "total_operations", "avg_span_duration"]

    # Ensure features exist in df_test
    missing_features = [col for col in feature_columns if col not in df_test.columns]
    if missing_features:
        print(f"Error: Missing features in test data: {missing_features}")
        return

    # Extract feature matrix
    X_test = df_test[feature_columns].fillna(0).values

    # Standardize features (assuming the model was trained with standardized data)
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    # Predict labels
    predictions = model.predict(X_test_scaled)

    # Map numerical labels back to original class names
    label_mapping = {0: "Normal", 1: "Blob", 2: "EST", 3: "Chatty"}
    df_test["predicted_label"] = [label_mapping[p] for p in predictions]

    # Print predictions
    print("\nPredictions on New Data:")
    print(df_test[["trace_id", "service_name", "predicted_label"]].head(10))  # Show first 10 results

    return df_test

def compute_graph_centralities(df_request, sample_fraction=0.1):
    """
    Compute approximate centrality measures for the services in df_request using sampling.
    """
    G = nx.DiGraph()

    # Extract pairwise service interactions within each request
    service_pairs = df_request.groupby(["trace_id", "service_name"]).size().reset_index(name="interaction_count")

    # Add edges with weights (interaction frequency)
    for _, row in service_pairs.iterrows():
        G.add_edge(row["trace_id"], row["service_name"], weight=row["interaction_count"])

    # Compute degree centrality first (cheap operation)
    degree_centrality = nx.degree_centrality(G)

    # **Optimized Step: Compute betweenness only on a sample of nodes**
    sampled_nodes = list(df_request["service_name"].drop_duplicates().sample(frac=sample_fraction))
    betweenness_centrality = nx.betweenness_centrality(G, k=sampled_nodes, weight="weight")

    # Map computed betweenness back to df_request
    df_request["betweenness"] = df_request["service_name"].map(betweenness_centrality).fillna(0)
    df_request["degree"] = df_request["service_name"].map(degree_centrality).fillna(0)

    return df_request


def compute_causality_scores(df_request):
    """
    Computes per-request causality scores.
    """
    causality_scores = {}

    for trace_id in df_request["trace_id"].unique():
        df_trace = df_request[df_request["trace_id"] == trace_id]

        for source in df_trace["service_name"]:
            for target in df_trace["service_name"]:
                if source != target:
                    try:
                        source_series = df_trace[df_trace["service_name"] == source]["span_duration"].values
                        target_series = df_trace[df_trace["service_name"] == target]["span_duration"].values

                        if len(source_series) > 1 and len(target_series) > 1:
                            test_result = grangercausalitytests(
                                np.column_stack([source_series, target_series]), maxlag=1, verbose=False
                            )
                            p_value = test_result[1][0]["ssr_ftest"][1]
                            causality_strength = 1 - p_value if p_value < 0.05 else 0

                            causality_scores[(trace_id, source, target)] = causality_strength
                    except:
                        continue

    return causality_scores

def construct_causal_graph(df_request, causality_scores):
    """
    Construct a weighted causal graph using per-request data.
    """
    # Ensure betweenness is computed
    if "betweenness" not in df_request.columns:
        print("Betweenness centrality missing. Recomputing...")
        df_request = compute_graph_centralities(df_request)

    G_causal = nx.DiGraph()

    # Compute Root Cause Score per request
    df_request["root_cause_score"] = df_request.apply(
        lambda row: sum(
            causality_scores.get((row["service_name"], target), 0)
            for target in df_request["service_name"]
        ) + row["betweenness"] * 0.2 + row["error_count"] * 0.1,
        axis=1,
    )

    return df_request.sort_values(by=["trace_id", "root_cause_score"], ascending=[True, False])

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
    input_folder = sys.argv[1]
    test_data_path = sys.argv[2]  # Third input: New Data File for Prediction
    df = process_all_csv_in_folder(input_folder)
    df= optimize_memory(df)

    # **1. Compute Per-Request Features**
    df_request = process_request_level_data(df)

    # **2. Compute Weak Labels**
    df_request = compute_weak_labels(df_request)

    # **3. Train Classification Model**
    model = train_classification_model(df_request)

    # **4. Compute Causality Scores**
    causality_scores = compute_causality_scores(df_request)

    # **5. Construct Causal Graph**
    df_request = construct_causal_graph(df_request, causality_scores)

    # Save Results
    df_request.to_csv("df_cause_sorted.csv", index=False)

    # Process new test data
    df_test = pd.read_csv(test_data_path)
    df_test = optimize_memory(df_test)

    # **1. Compute Per-Request Features**
    df_test = process_request_level_data(df_test)

    # **2. Compute Weak Labels**
    df_test = compute_weak_labels(df_test)

    # Predict using the trained model
    df_test = load_model_and_predict(model, df_test)

    # Save results
    df_test.to_csv("predicted_results.csv", index=False)
