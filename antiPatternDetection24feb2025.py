
from scipy.stats import mode
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import os
import matplotlib.pyplot as plt
import sys
import networkx as nx
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import matplotlib.cm as cm


def compute_weak_label_scores(df_request):
    """
    Compute weak label scores per request with fine-tuned scoring based on weighted features.
    """
    thresholds = {
        "service_contribution": df_request.get("service_contribution", pd.Series([0])).quantile(0.75),
        "total_operations": df_request.get("total_operations", pd.Series([0])).quantile(0.75),
        "span_duration": df_request.get("span_duration", pd.Series([0])).quantile(0.75),
        "send_syscall_count": df_request.get("send_syscall_count", pd.Series([0])).quantile(0.75),
        "transmitted_data_per_call": (
            df_request.get("transmitted_data", pd.Series([0]))
            / df_request.get("send_syscall_count", pd.Series([1]))
        ).quantile(0.25),
    }

    df_request["Blob_score"] = 0.6 * (
        df_request.get("service_contribution", pd.Series([0])) / thresholds["service_contribution"]
    ) + 0.4 * (
        df_request.get("total_operations", pd.Series([0])) / thresholds["total_operations"]
    )

    df_request["Chatty_score"] = 0.5 * (
        df_request.get("total_operations", pd.Series([0])) / thresholds["total_operations"]
    ) + 0.5 * (
        df_request.get("span_duration", pd.Series([0])) / thresholds["span_duration"]
    )

    df_request["EST_score"] = 0.7 * (
        df_request.get("send_syscall_count", pd.Series([0])) / thresholds["send_syscall_count"]
    ) + 0.3 * (
        thresholds["transmitted_data_per_call"]
        / (
            df_request.get("transmitted_data", pd.Series([0]))
            / df_request.get("send_syscall_count", pd.Series([1]))
        )
    )

    features = [
        "span_duration",
        "recv_data",
        "transmitted_data",
        "recv_syscall_count",
        "send_syscall_count",
        "total_operations",
        "avg_size_recv_msg",
        "avg_size_send_msg",
        "service_contribution",
        "error_count",
    ]

    # Ensure Normal_score is consistent
    df_request["Normal_score"] = 1 - (
        df_request[["Blob_score", "Chatty_score", "EST_score"]].max(axis=1)
    )

    return df_request


def kmeans_clustering(df_request, n_clusters=6):
    """
    Apply K-Means Clustering, including a 'Normal' cluster, and return cluster labels.
    Ensures weak label scores are computed if missing.
    """
    required_scores = ["Blob_score", "Chatty_score", "EST_score", "Normal_score"]
    if not all(score in df_request.columns for score in required_scores):
        print("\n⚠️ Weak label scores missing in data. Computing them now...")
        df_request = compute_weak_label_scores(df_request)

    features = [
        "span_duration",
        "recv_data",
        "transmitted_data",
        "recv_syscall_count",
        "send_syscall_count",
        "total_operations",
        "avg_size_recv_msg",
        "avg_size_send_msg",
        "service_contribution",
        "error_count",
    ]

    available_features = [f for f in features if f in df_request.columns]
    X = df_request[available_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_request["cluster_label"] = kmeans.fit_predict(X_scaled)

    for i in range(n_clusters):
        cluster_name = "Normal" if i == 0 else f"Cluster_{i}"
        df_request[f"{cluster_name}_score"] = (
            df_request["cluster_label"] == i
        ).astype(int)

    return df_request, kmeans


def refine_labels_with_scores(df_request, kmeans, weight_weak=0.6, weight_cluster=0.4):
    """
    Combine weak label scores and K-Means cluster membership scores to finalize labels,
    ensuring dimensions match.
    """
    n_clusters = kmeans.n_clusters
    cluster_names = ["Normal"] + [f"Cluster_{i}" for i in range(1, n_clusters)]

    # Ensure all required cluster score columns exist
    for name in cluster_names:
        if f"{name}_score" not in df_request.columns:
            df_request[f"{name}_score"] = 0

    cluster_scores = df_request[[f"{name}_score" for name in cluster_names]].values
    weak_scores = df_request[[
        "Blob_score", "Chatty_score", "EST_score", "Normal_score"
    ]].values

    # Adjust weak_scores if needed
    if weak_scores.shape[1] < cluster_scores.shape[1]:
        padding = np.zeros((weak_scores.shape[0], cluster_scores.shape[1] - weak_scores.shape[1]))
        weak_scores = np.hstack((weak_scores, padding))

    combined_scores = (
        weight_weak * weak_scores + weight_cluster * cluster_scores
    )

    label_names = ["Blob", "Chatty", "EST", "Normal"] + [
        f"Cluster_{i}" for i in range(1, n_clusters)
    ]
    final_labels = [label_names[np.argmax(row)] for row in combined_scores]

    df_request["final_label"] = final_labels

    print("\n🔹 Final Label Distribution:")
    print(df_request["final_label"].value_counts())

    return df_request


def classify_with_kfold(df_train, df_test, n_splits=5):
    """
    Train and evaluate a classifier using separate training and testing datasets
    with k-fold validation and report performance metrics.
    Automatically generates 'final_label' and weak scores for test data if missing.
    """
    if "final_label" not in df_test.columns:
        print("\n⚠️ 'final_label' not found in test data. Generating labels...")
        df_test, kmeans_test = kmeans_clustering(df_test)
        df_test = refine_labels_with_scores(df_test, kmeans_test)

    features = [
        "span_duration",
        "recv_data",
        "transmitted_data",
        "recv_syscall_count",
        "send_syscall_count",
        "total_operations",
        "avg_size_recv_msg",
        "avg_size_send_msg",
        "service_contribution",
        "error_count",
    ]

    available_features = [f for f in features if f in df_train.columns]

    X_train = df_train[available_features].fillna(0).values
    y_train = df_train["final_label"].values
    X_test = df_test[available_features].fillna(0).values
    y_test = df_test["final_label"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = RandomForestClassifier(random_state=42)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracy = cross_val_score(classifier, X_train_scaled, y_train, cv=kf, scoring="accuracy").mean()
    precision = cross_val_score(classifier, X_train_scaled, y_train, cv=kf, scoring="precision_weighted").mean()
    recall = cross_val_score(classifier, X_train_scaled, y_train, cv=kf, scoring="recall_weighted").mean()
    f1 = cross_val_score(classifier, X_train_scaled, y_train, cv=kf, scoring="f1_weighted").mean()

    print("\n🔹 K-Fold Validation Results on Training Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)

    print("\n🔹 Evaluation on Test Set:")
    print(classification_report(y_test, y_pred))

    return accuracy, precision, recall, f1

def visualize(df_request):
    # Ensure required columns exist
    cluster_col = "cluster_based_label"
    tree_col = "tree_refined_label"
    final_col = "final_refined_label"
    features = [
        "span_duration", "recv_data", "transmitted_data", "recv_syscall_count",
        "send_syscall_count", "total_operations", "avg_size_recv_msg", "avg_size_send_msg",
        "ratio_recv_time", "ratio_sent_time", "service_contribution", "error_count"
    ]

    # Filter available features for visualization
    available_features = [f for f in features if f in df_request.columns]

    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_request[available_features].fillna(0))

    df_request["PCA1"] = X_pca[:, 0]
    df_request["PCA2"] = X_pca[:, 1]

    # Define color palette for labels
    palette = {
        "Blob": "red", "Normal": "orange", "EST": "green", "Chatty": "blue"
    }

    # --- 1️⃣ K-Means Clustering Visualization ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df_request["PCA1"], y=df_request["PCA2"], hue=df_request[cluster_col],
        palette=palette, alpha=0.6, edgecolor="k"
    )
    plt.title("K-Means Clustering Results (PCA Reduced)")
    plt.legend(title="Cluster Label")
    plt.savefig('Cluster_label.png', dpi=300)

    # --- 2️⃣ Decision Tree Classification Visualization ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df_request["PCA1"], y=df_request["PCA2"], hue=df_request[tree_col],
        palette=palette, alpha=0.6, edgecolor="k"
    )
    plt.title("Decision Tree Classification Results (PCA Reduced)")
    plt.legend(title="Tree Refined Label")
    plt.savefig('tree_label.png', dpi=300)


def feature_definition(df_train):
    # Features to use for training
    features = [
        "span_duration", "recv_data", "transmitted_data", "recv_syscall_count",
        "send_syscall_count", "total_operations", "avg_recv_msg_size", "avg_sent_msg_size",
        "ratio_recv_time", "ratio_sent_time", "service_contribution", "error_count"
    ]

    available_features = [f for f in features if f in df_train.columns]

    # Extract features (X) and labels (y) for training
    X_train = df_train[available_features].fillna(0)
    y_train = df_train["final_refined_label"]  # Use final refined labels for training

    # Encode categorical labels to numeric
    label_mapping = {"Normal": 0, "Blob": 1, "EST": 2, "Chatty": 3}
    y_train = y_train.map(label_mapping)

    # ✅ Scale features using StandardScaler (Fit on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fitting happens here!

    # Train a Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    rf_model.fit(X_train_scaled, y_train)

    # Check training accuracy
    train_acc = rf_model.score(X_train_scaled, y_train)
    print(f"Training Accuracy: {train_acc:.4f}")
    return available_features, X_train_scaled, y_train, scaler

def train_classifier(X_train_scaled, y_train, model_path="model.joblib", scaler_path="scaler.joblib"):
    # Train a Random Forest Classifier
    # rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    rf_model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Check training accuracy
    train_acc = rf_model.score(X_train_scaled, y_train)
    print(f"Training Accuracy: {train_acc:.4f}")

    # Save model and scaler
    joblib.dump(rf_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    return rf_model


def predict_labels(df_test, model_path="model.joblib", scaler_path="scaler.joblib"):
    """
    Predict labels for the test dataset using the trained model.
    """
    features = [
        "span_duration", "recv_data", "transmitted_data", "recv_syscall_count",
        "send_syscall_count", "total_operations", "avg_recv_msg_size", "avg_sent_msg_size",
        "ratio_recv_time", "ratio_sent_time", "service_contribution", "error_count"
    ]

    available_features = [f for f in features if f in df_test.columns]
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X_test = df_test[available_features].fillna(0)  # Extract features from test data
    X_test_scaled = scaler.transform(X_test)  # Scale the features

    # Predict Labels
    y_pred = model.predict(X_test_scaled)

    # Map numeric labels back to original categories
    reverse_mapping = {0: "Normal", 1: "Blob", 2: "EST", 3: "Chatty"}
    df_test["predicted_label"] = [reverse_mapping[label] for label in y_pred]
    print("Predictions complete.")
    return df_test


def compute_per_trace_durations(df):
    """
    Recompute sum_span_duration at the per-trace level instead of aggregated service level.
    """
    df_trace_duration = df.groupby("trace_id")["span_duration"].sum().reset_index()
    df_trace_duration.rename(columns={"span_duration": "sum_span_duration"}, inplace=True)

    # Merge back to df
    df = df.merge(df_trace_duration, on="trace_id", how="left")
    return df

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
    visualize_causal_graph(G_causal, df_service)

    return df_service.sort_values(by="root_cause_score", ascending=False)

def visualize_causal_graph(G, df_service):
    """
    Draws a directed causal graph with edge weights and color-coded nodes based on root cause scores.
    """
    # Use spring layout with spacing
    pos = nx.spring_layout(G, k=1.5, iterations=100)

    # Make sure service_scores has unique indices
    service_scores = df_service.drop_duplicates(subset=["service_name"]).set_index("service_name")["root_cause_score"]

    # Get float scores safely
    scores = [float(service_scores.get(node, 0)) for node in G.nodes()]
    norm = plt.Normalize(min(scores), max(scores))
    if not scores:
        raise ValueError("No root cause scores available to visualize.")

    cmap = cm.get_cmap('coolwarm')
    node_colors = [cmap(norm(score)) for score in scores]

    # Edge labels (weights)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    # Plot
    plt.figure(figsize=(12, 9))
    nx.draw(G, pos,
            with_labels=True,
            node_color=node_colors,
            edge_color='gray',
            node_size=2200,
            font_size=10,
            arrowsize=20,
            width=1.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Root Cause Score")

    plt.title("Causal Graph with Root Cause Highlighting", fontsize=14)
    plt.tight_layout()
    plt.savefig("Causal_Graph_Color.png", dpi=300)
    plt.show()

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

def plot_feature_distributions(df):
    feature_columns = [
        'span_duration', 'recv_data', 'transmitted_data', 'recv_syscall_count',
        'send_syscall_count', 'total_operations', 'avg_size_recv_msg',
        'avg_size_send_msg', 'service_contribution'
    ]

    plt.figure(figsize=(20, 20))
    for i, feature in enumerate(feature_columns, 1):
        plt.subplot(4, 3, i)
        if feature in df.columns:
            data = df[feature].dropna()

            # Clip extreme outliers for better visibility
            q_low = data.quantile(0.001)
            q_high = data.quantile(0.999)
            clipped_data = data[(data >= q_low) & (data <= q_high)]

            # Use log scale if data is highly skewed
            use_log = (clipped_data.skew() > 2 and clipped_data.min() > 0)

            sns.histplot(clipped_data, kde=True, bins=50, color='blue')
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.grid(True)
            if use_log:
                plt.xscale("log")

    plt.tight_layout()
    plt.savefig("feature_distribution_adaptive.png", dpi=300)
    plt.show()

def plot_scatter_by_trace_type(df):
    if {'trace_type', 'span_duration', 'total_operations'}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='span_duration', y='total_operations', hue='trace_type', data=df, palette='coolwarm')
        plt.title('Scatter Plot: Span Duration vs. Total Operations by Trace Type')
        plt.savefig('scatter_plot_trace_types_updated.png', dpi=300)

def plot_hexbin_scatter_by_trace_type(df):
    if {'span_duration', 'total_operations'}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        plt.hexbin(df['span_duration'], df['total_operations'], gridsize=50, cmap='Blues', mincnt=1)
        plt.colorbar(label='Density')
        plt.title('Hexbin Scatter Plot: Span Duration vs. Total Operations')
        plt.xlabel('Span Duration')
        plt.ylabel('Total Operations')
        plt.savefig('hexbin_scatter_plot_trace_types_updated.png', dpi=300)

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

    # Use Graphviz layout for better label spacing
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')  # requires pygraphviz
    except:
        pos = nx.spring_layout(G, k=1.2, iterations=100)

    plt.figure(figsize=(14, 12))
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color='lightblue',
            font_size=9, font_weight='bold', edge_color='gray', arrowsize=15)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)

    plt.title('Service Call Diagram for Entire Dataset')
    plt.tight_layout()
    plt.savefig('service_call_network_updated.png', dpi=300)
    plt.show()


# Define a focused plotting function for selected features
def plot_focused_feature_distributions(df):
    feature_columns = [
        'span_duration', 'recv_data', 'transmitted_data', 'recv_syscall_count','send_syscall_count',
                                      'total_operations', 'avg_size_recv_msg',
    'avg_size_send_msg', 'service_contribution'
    ]

    plt.figure(figsize=(20, 16))
    for i, feature in enumerate(feature_columns, 1):
        plt.subplot(3, 3, i)
        if feature in df.columns:
            data = df[feature].dropna()
            data = data[data != 0]
            # Adaptive clipping and log scale
            q_low = data.quantile(0.001)
            q_high = data.quantile(0.995)
            clipped_data = data[(data >= q_low) & (data <= q_high)]

            use_log = (clipped_data.skew() > 2 and clipped_data.min() > 0)

            sns.histplot(clipped_data, kde=True, bins=50, color='steelblue')
            plt.title(f"Focused Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.grid(True)
            if use_log:
                plt.xscale("log")

    plt.tight_layout()
    plt.savefig("focused_feature_distribution.png", dpi=300)

def plot_boxplot_and_pairplot(df, feature, max_samples=1000):

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='final_label', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Anti-Pattern Label')
    plt.savefig('boxplot_antipattern_label.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Sampling to reduce memory usage
    if len(df) > max_samples:
        df_sampled = df.sample(n=max_samples, random_state=42)
    else:
        df_sampled = df.copy()

    # Reduce dimensionality of features for faster pairplot
    pairplot_features = [
        'span_duration', 'recv_data', 'transmitted_data', 'recv_syscall_count', 'send_syscall_count',
        'total_operations', 'avg_size_recv_msg',
        'avg_size_send_msg', 'service_contribution'
    ]
    available_features = [f for f in pairplot_features if f in df_sampled.columns]

    pairplot_fig = sns.pairplot(df_sampled, hue='final_label',
                                 vars=available_features, diag_kind='hist', corner=True)
    pairplot_fig.fig.suptitle('Pairplot of Selected Features by Anti-Pattern Label', y=1.02)
    pairplot_fig.savefig('pairplot_antipattern_label.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_boxplot_and_pairplot_test(df, feature, max_samples=1000):

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='final_label', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Anti-Pattern Label')
    plt.savefig('boxplot_antipattern_label_test.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Sampling to reduce memory usage
    if len(df) > max_samples:
        df_sampled = df.sample(n=max_samples, random_state=42)
    else:
        df_sampled = df.copy()

    # Reduce dimensionality of features for faster pairplot
    pairplot_features = [
        'span_duration', 'recv_data', 'transmitted_data', 'recv_syscall_count', 'send_syscall_count',
        'total_operations', 'avg_size_recv_msg',
        'avg_size_send_msg', 'service_contribution'
    ]
    available_features = [f for f in pairplot_features if f in df_sampled.columns]

    pairplot_fig = sns.pairplot(df_sampled, hue='final_label',
                                 vars=available_features, diag_kind='hist', corner=True)
    pairplot_fig.fig.suptitle('Pairplot of Selected Features by Anti-Pattern Label', y=1.02)
    pairplot_fig.savefig('pairplot_antipattern_label_test.png', dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    df = process_all_csv_in_folder(train_data)
    df = optimize_memory(df)
    df_test = process_all_csv_in_folder(test_data)
    df_test = optimize_memory(df_test)

    # **Step 0: Compute Weak Labels Per Request**
    df_request = compute_weak_label_scores(df)
    df_request, kmeans= kmeans_clustering(df_request, 5)
    refine_labels_with_scores(df_request, kmeans, weight_weak=0.6, weight_cluster=0.4)
    classify_with_kfold(df_request, df_test, n_splits=5)

    ## now use classification to classifiy the data using the provided labels
    ## use k-fold validation to validate
    ## report accuracy, percesion, recall and f1-measure

    print(df_request.groupby("service_name")["span_duration"].nunique())
    df_request["final_refined_label"] = df_request["final_label"]
    available_features, X_train_scaled, y_train, scaler = feature_definition(df_request)
  # Predict on the test dataset
    df_test = predict_labels(df_test)

    # Optional visualizations for deeper analysis
    # plot_feature_distributions(df_request)
    # plot_scatter_by_trace_type(df_request)
    # plot_hexbin_scatter_by_trace_type(df_request)
    # plot_service_call_network(df_request)
    # plot_focused_feature_distributions(df_request)
    plot_boxplot_and_pairplot_test(df_test, 'span_duration')
    plot_boxplot_and_pairplot(df_request, 'span_duration')

    # Show predictions
    print(df_test[["service_name", "op_name", "predicted_label"]].head())

    # Save predictions to a CSV file
    df_test.to_csv("predicted_test_labels.csv", index=False)
    print("Predictions saved to 'predicted_test_labels.csv'.")
    # evalute_classifier(df_test)


    # **Step 2: Merge back with Original Data**
    df_test_spa = df_test.merge(df_test[["trace_id"]], on="trace_id", how="inner")

    # **Step 3: Restore Per-Trace Span Duration**
    df_test_spa = compute_per_trace_durations(df_test_spa)
    # **Step 4: Compute Granger Causality**
    causality_scores = compute_granger_causality(df_test_spa)
    print("Granger Causality Scores Sample:", list(causality_scores.items())[:5])
    # **Step 5: Compute Trace-Level Centrality**
    df_spa = compute_trace_level_centrality(df_test_spa)

    # **Step 6: Construct Root Cause Graph**
    # df_root_cause = construct_causal_graph(df_test_spa, causality_scores)
    df_root_cause = construct_causal_graph(df_spa, causality_scores)

    # Assign predicted labels as 'final_refined_label'
    df_root_cause["final_refined_label"] = df_test.set_index(["service_name", "op_name"]).loc[
        df_root_cause.set_index(["service_name", "op_name"]).index
    ]["predicted_label"].values

    print(df_test_spa[["betweenness", "degree", "closeness"]].describe())
    df_root_cause.to_csv("df_root_cause.csv", index=False)
    df_root_cause_filtered = df_root_cause[df_root_cause["final_refined_label"].isin(["Blob", "EST", "Chatty"])]

    df_root_cause_unique = df_root_cause.drop_duplicates(subset=["service_name", "op_name"], keep="first")
    df_root_cause_unique.to_csv("df_root_cause_unique.csv", index=False)
    df_root_cause_filtered = df_root_cause_unique[df_root_cause_unique["final_refined_label"].isin(["Blob", "EST", "Chatty"])]
    df_root_cause_filtered.to_csv("df_root_cause_filtered.csv", index=False)
    df_test_unique = df_test.drop_duplicates(subset=["service_name", "op_name"], keep="first")
    df_test_unique.to_csv("df_test_cause_unique.csv", index=False)

    model = train_classifier(X_train_scaled, y_train)


