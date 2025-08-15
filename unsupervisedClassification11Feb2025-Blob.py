from statsmodels.tsa.stattools import grangercausalitytests
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import jaccard_score, silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import numpy as np

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering

# Example: Features specific to each class
FEATURES = {
    'Blob': ['Source_Service', 'Target_Service', 'syscall_Count',
               'Request_Intensity', 'Avg_Data_Sent', 'Avg_Data_Received', 'Error_Rate']
# 'Request_Intensity', 'Avg_Data_Sent', 'Avg_Data_Received', 'Error_Rate', 'betweenness', 'degree','closeness']
}

def create_knn_anomaly_pipeline(k=2):
    """
    Uses KNN for anomaly detection based on distance to nearest neighbors.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('knn_anomaly', KNNAnomalyDetector(k=k))
    ])

def create_minibatch_kmeans_pipeline():
    """
    Creates a MiniBatchKMeans clustering pipeline (fast clustering for large datasets).
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=100))
    ])


def create_gaussian_mixture_pipeline():
    """
    Creates a Gaussian Mixture Model (GMM) pipeline for soft clustering (probabilistic assignment).
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', GaussianMixture(n_components=2, random_state=42))
    ])


def create_hierarchical_clustering_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.9)),  # Retains 90% of variance
        ('model', AgglomerativeClustering(n_clusters=2))
    ])


class KNNAnomalyDetector:
    """
    Custom KNN-based anomaly detection model.
    """

    def __init__(self, k=2):
        self.k = k
        self.knn = NearestNeighbors(n_neighbors=self.k)
        self.anomaly_scores_ = None

    def fit(self, X, y=None):
        X = StandardScaler().fit_transform(X)
        self.knn.fit(X)
        distances, _ = self.knn.kneighbors(X)
        self.anomaly_scores_ = np.mean(distances, axis=1)  # Higher distance = more anomalous
        return self

    def predict(self, X):
        X = StandardScaler().fit_transform(X)
        distances, _ = self.knn.kneighbors(X)
        scores = np.mean(distances, axis=1)
        return (scores > np.percentile(self.anomaly_scores_, 95)).astype(int)  # Flag top 5% as anomalies


def train_models(train_df, FEATURES):
    models = {}
    print(f"\n[INFO] Starting model training...")
    print(f"[INFO] Training dataset shape: {train_df.shape}\n")

    for category, features in FEATURES.items():
        print(f"[INFO] Processing category: {category}")
        X_train = train_df[features].copy()

        for col in ["Source_Service", "Target_Service"]:
            if col in X_train.columns:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col])
                print(f"[INFO] Encoded {col}. Unique classes: {len(le.classes_)}")

        print(f"[INFO] Feature set shape for '{category}': {X_train.shape}")
        print(f"[INFO] Data types after encoding:\n{X_train.dtypes}\n")

        models[category] = {
            'MiniBatchKMeans': create_minibatch_kmeans_pipeline(),
            'GaussianMixture': create_gaussian_mixture_pipeline(),
            # 'AgglomerativeClustering': create_hierarchical_clustering_pipeline(),
            'KNNAnomalyDetection': create_knn_anomaly_pipeline(k=2)  # Using KNN for anomaly detection
        }

        for model_name, model in models[category].items():
            print(f"[INFO] Training {model_name} for category '{category}'...")
            model.fit(X_train)
            print(f"[SUCCESS] {model_name} trained successfully for '{category}'!\n")

    print(f"[COMPLETE] All models trained!\n")
    return models

def predict_clusters(test_df, models, FEATURES):
    predictions = {}
    print(f"\n[INFO] Starting predictions...")
    print(f"[INFO] Test dataset shape: {test_df.shape}\n")

    all_outputs = {}

    for category, model_dict in models.items():
        X_test = test_df[FEATURES[category]].copy()

        for col in ["Source_Service", "Target_Service"]:
            if col in X_test.columns:
                le = LabelEncoder()
                X_test[col] = le.fit_transform(X_test[col])
                print(f"[INFO] Encoded {col} for predictions. Unique classes: {len(le.classes_)}")

        for model_name, model in model_dict.items():
            print(f"[INFO] Predicting with {model_name} for '{category}'...")
            pred = model.predict(X_test)
            all_outputs[(category, model_name)] = pred

            try:
                if model_name == "KNNAnomalyDetection":
                    X_scaled = model.named_steps['scaler'].transform(X_test)
                    knn_model = model.named_steps['knn_anomaly']
                    distances, _ = knn_model.knn.kneighbors(X_scaled)
                    scores = np.mean(distances, axis=1)
                    all_outputs[(category, f"{model_name}_scores")] = scores

                elif model_name == "MiniBatchKMeans":
                    X_scaled = model.named_steps['scaler'].transform(X_test)
                    centers = model.named_steps['model'].cluster_centers_
                    distances = pairwise_distances(X_scaled, centers)
                    min_dist = np.min(distances, axis=1)
                    all_outputs[(category, f"{model_name}_scores")] = min_dist

                elif model_name == "GaussianMixture":
                    X_scaled = model.named_steps['scaler'].transform(X_test)
                    scores = -model.named_steps['model'].score_samples(X_scaled)
                    all_outputs[(category, f"{model_name}_scores")] = scores

            except Exception as e:
                print(f"[WARN] Could not compute scores for {model_name} in {category}: {e}")

    print(f"[COMPLETE] All predictions completed!\n")

    # 🔐 SAFEGUARD: Validate that all outputs match test_df length
    for key, value in all_outputs.items():
        if len(value) != len(test_df):
            raise ValueError(f"[ERROR] Prediction length mismatch for {key}: got {len(value)}, expected {len(test_df)}")

    # ✅ Safe to return now
    return pd.DataFrame(all_outputs, index=test_df.index)


#
# def predict_clusters(test_df, models, FEATURES):
#     predictions = {}
#     print(f"\n[INFO] Starting predictions...")
#     print(f"[INFO] Test dataset shape: {test_df.shape}\n")
#
#     for category, model_dict in models.items():
#         X_test = test_df[FEATURES[category]].copy()
#
#         for col in ["Source_Service", "Target_Service"]:
#             if col in X_test.columns:
#                 le = LabelEncoder()
#                 X_test[col] = le.fit_transform(X_test[col])
#                 print(f"[INFO] Encoded {col} for predictions. Unique classes: {len(le.classes_)}")
#
#         predictions[category] = {}
#         for model_name, model in model_dict.items():
#             print(f"[INFO] Predicting with {model_name} for '{category}'...")
#             pred = model.predict(X_test)
#             predictions[category][model_name] = pred
#
#     print(f"[COMPLETE] All predictions completed!\n")
#     return pd.DataFrame(
#         {(cat, model): preds for cat, model_preds in predictions.items() for model, preds in model_preds.items()},
#         index=test_df.index)
#
#
# def train_models(train_df, FEATURES):
#     models = {}
#     print(f"\n[INFO] Starting model training...")
#     print(f"[INFO] Training dataset shape: {train_df.shape}\n")
#
#     for category, features in FEATURES.items():
#         print(f"[INFO] Processing category: {category}")
#         X_train = train_df[features].copy()
#
#         for col in ["Source_Service", "Target_Service"]:
#             if col in X_train.columns:
#                 le = LabelEncoder()
#                 X_train[col] = le.fit_transform(X_train[col])
#                 print(f"[INFO] Encoded {col}. Unique classes: {len(le.classes_)}")
#
#         print(f"[INFO] Feature set shape for '{category}': {X_train.shape}")
#         print(f"[INFO] Data types after encoding:\n{X_train.dtypes}\n")
#
#         models[category] = {
#             'MiniBatchKMeans': create_minibatch_kmeans_pipeline(),
#             'GaussianMixture': create_gaussian_mixture_pipeline()
#             # 'HierarchicalClustering': create_hierarchical_clustering_pipeline()
#             # 'AgglomerativeClustering': create_hierarchical_clustering_pipeline()  # If you prefer hierarchy
#
#         }
#
#         for model_name, model in models[category].items():
#             print(f"[INFO] Training {model_name} for category '{category}'...")
#             model.fit(X_train)
#             print(f"[SUCCESS] {model_name} trained successfully for '{category}'!\n")
#
#     print(f"[COMPLETE] All models trained!\n")
#     return models
#
#
# def predict_clusters(test_df, models, FEATURES):
#     predictions = {}
#     print(f"\n[INFO] Starting predictions...")
#     print(f"[INFO] Test dataset shape: {test_df.shape}\n")
#
#     for category, model_dict in models.items():
#         X_test = test_df[FEATURES[category]].copy()
#
#         for col in ["Source_Service", "Target_Service"]:
#             if col in X_test.columns:
#                 le = LabelEncoder()
#                 X_test[col] = le.fit_transform(X_test[col])
#                 print(f"[INFO] Encoded {col} for predictions. Unique classes: {len(le.classes_)}")
#
#         predictions[category] = {}
#         for model_name, model in model_dict.items():
#             print(f"[INFO] Predicting with {model_name} for '{category}'...")
#             pred = model.predict(X_test)
#             predictions[category][model_name] = pred
#
#     print(f"[COMPLETE] All predictions completed!\n")
#     return pd.DataFrame({(cat, model): preds for cat, model_preds in predictions.items() for model, preds in model_preds.items()}, index=test_df.index)
#

def calculate_fuzzy_membership(model, X):
    """
    Calculates fuzzy membership values for MiniBatchKMeans clusters.

    Args:
        model: Trained MiniBatchKMeans model.
        X: Scaled input data.

    Returns:
        np.ndarray: Membership matrix.
    """
    distances = pairwise_distances(X, model.cluster_centers_)
    inverted = 1 / (distances + 1e-6)
    return inverted / inverted.sum(axis=1, keepdims=True)
# def compute_trace_level_centrality(df_service):
#     """
#     Compute betweenness, degree, and closeness centrality per trace.
#     """
#     G = nx.DiGraph()
#
#     # Extract service interactions per trace
#     # service_pairs = df_service.groupby(["Trace_ID", "Source_Service"])["syscall_Count"].sum().reset_index()
#
#     for _, row in df_service.iterrows():
#         G.add_edge(row["Source_Service"], row["Target_Service"], weight=row["syscall_Count"])
#
#     # Compute per-trace centrality
#     betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
#     degree = nx.degree_centrality(G)
#     closeness = nx.closeness_centrality(G)
#
#     for node in G.nodes:
#         print(
#             f"Node {node}: Degree={degree[node]}, Closeness={closeness[node]}, Betweenness={betweenness[node]}")
#
#     # Assign metrics to dataframe
#     df_service["betweenness"] = df_service["Trace_ID"].map(betweenness)
#     df_service["degree"] = df_service["Trace_ID"].map(degree)
#     df_service["closeness"] = df_service["Trace_ID"].map(closeness)
#
#     return df_service

def compute_trace_level_centrality(df_service):
    """
    Compute betweenness, degree, and closeness centrality per trace.
    For each Trace_ID, constructs a subgraph and computes centrality measures
    based on Source_Service and Target_Service pairs within that trace.
    """
    trace_centrality_results = []

    for trace_id, trace_group in df_service.groupby("Trace_ID"):
        G = nx.DiGraph()
        for _, row in trace_group.iterrows():
            G.add_edge(row["Source_Service"], row["Target_Service"], weight=row["syscall_Count"])

        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        degree = nx.degree_centrality(G)
        closeness = nx.closeness_centrality(G)

        for node in G.nodes:
            trace_centrality_results.append({
                "Trace_ID": trace_id,
                "Service": node,
                "betweenness": betweenness.get(node, 0),
                "degree": degree.get(node, 0),
                "closeness": closeness.get(node, 0)
            })

    df_centrality = pd.DataFrame(trace_centrality_results)

    # Merge computed centralities back to the original DataFrame based on Trace_ID and Source_Service
    df_service = df_service.merge(
        df_centrality,
        left_on=["Trace_ID", "Source_Service"],
        right_on=["Trace_ID", "Service"],
        how="left"
    ).drop(columns=["Service"])

    return df_service

def visualize_prediction_overlap(if_preds, svm_preds):
    overlap = np.mean(if_preds == svm_preds)
    print(f"Prediction overlap between IsolationForest and OneClassSVM: {overlap:.2%}")

    plt.bar(['Matching Predictions', 'Differing Predictions'],
            [sum(if_preds == svm_preds), sum(if_preds != svm_preds)],
            color=['green', 'red'])
    plt.title(f"Prediction Overlap: {overlap:.2%}")
    plt.ylabel("Count of Samples")
    plt.savefig('Prediction_Overlap.png', dpi=300)


def visualize_svm_scores(model, X_test_scaled):
    scores = model.decision_function(X_test_scaled)
    plt.hist(scores, bins=50, color='orange', alpha=0.7)
    plt.title("OneClassSVM Decision Function Distribution")
    plt.xlabel("Decision Function Score")
    plt.ylabel("Frequency")
    plt.savefig('OneClassSVM_Decision_Function_Distribution.png', dpi=300)

def visualize_isolationforest_scores(model, X_test_scaled):
    scores = model.decision_function(X_test_scaled)
    plt.hist(scores, bins=50, alpha=0.7)
    plt.title("Isolation Forest Anomaly Scores Distribution")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.savefig('Isolation_Forest_Anomaly_Scores_Distribution.png', dpi=300)

def visualize_scaled_data(X_train_scaled):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=pd.DataFrame(X_train_scaled))
    plt.title("Feature Distribution After Scaling")
    plt.xlabel("Features")
    plt.ylabel("Scaled Values")
    plt.savefig('Feature_Distribution_After_Scaling.png', dpi=300)


def visualize_anomaly_scores(models, predictions):
    """
    Visualizes the anomaly scores for each model if available.
    Matches keys based on the '{category}_{model_name}_scores' pattern.
    """
    for category, model_dict in models.items():
        for model_name, model in model_dict.items():
            score_key = (category, f"{model_name}_scores")
            if score_key in predictions.columns or score_key in predictions:
                # Handle MultiIndex or flat columns
                if score_key in predictions.columns:
                    scores = predictions[score_key].values
                else:
                    scores = predictions[score_key]

                plt.hist(scores, bins=50)
                plt.title(f"Anomaly Score Distribution - {category} - {model_name}")
                plt.xlabel("Anomaly Score")
                plt.ylabel("Frequency")
                plt.savefig(f'{category}_{model_name}_Anomaly_Score_Distribution.png', dpi=300)
                plt.close()
            else:
                print(f"[WARN] Scores for {model_name} in {category} not found in predictions.")

from sklearn.metrics import jaccard_score

def compare_models(pred1, pred2):
    similarity = jaccard_score(pred1, pred2)
    print(f"Jaccard Similarity between predictions: {similarity:.4f}")


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


def evaluate_silhouette_score_with_ground_truth(X, predicted_labels, true_labels):
    """
    Evaluates the clustering performance using the Silhouette Score considering ground truth labels.

    Args:
        X (pd.DataFrame or np.ndarray): The dataset used for clustering.
        predicted_labels (array-like): Cluster labels predicted by the model.
        true_labels (array-like): Ground truth labels for each sample.

    Returns:
        dict: Silhouette Scores for predicted clusters and true clusters.
    """
    predicted_score = silhouette_score(X, predicted_labels)
    true_score = silhouette_score(X, true_labels)

    print(f"Silhouette Score (Predicted Clusters): {predicted_score:.4f}")
    print(f"Silhouette Score (Ground Truth): {true_score:.4f}")

    return {"predicted_silhouette_score": predicted_score, "true_silhouette_score": true_score}


def service_pairwise_data(df):
    # Convert timestamp to a proper format if needed
    df['time_stamp'] = pd.to_numeric(df['time_stamp'])

    # Sort the data by trace_id and time_stamp to infer service interactions
    df_sorted = df.sort_values(by=['trace_id', 'time_stamp'])

    # Group by trace_id to establish service interactions
    pairwise_interactions = []
    for trace_id, group in df_sorted.groupby('trace_id'):
        services = group[
            ['service_name', 'recv_data', 'transmitted_data', 'recv_syscall_count', 'send_syscall_count', 'error_count',
             'span_duration']].values
        for i in range(len(services) - 1):
            source, recv_data_src, transmitted_data_src, recv_sys_src, send_sys_src, error_src, duration_src = services[
                i]
            target, recv_data_tgt, transmitted_data_tgt, recv_sys_tgt, send_sys_tgt, error_tgt, duration_tgt = services[
                i + 1]

            syscall_count = recv_sys_src + send_sys_src + recv_sys_tgt + send_sys_tgt
            request_count = 1  # Each interaction counts as one request
            avg_response_time = (duration_src + duration_tgt) / 2

            if avg_response_time > 0:
                request_intensity = request_count / avg_response_time
            else:
                request_intensity = 0

            avg_data_sent = (transmitted_data_src + transmitted_data_tgt) / 2
            avg_data_received = (recv_data_src + recv_data_tgt) / 2

            total_errors = error_src + error_tgt
            error_rate = total_errors / request_count if request_count > 0 else 0

            pairwise_interactions.append([group['time_stamp'].iloc[i], trace_id, source, target, syscall_count,
                                          request_intensity, avg_data_sent, avg_data_received, error_rate])

    # Create the final DataFrame
    columns = ['Timestamp', 'Trace_ID', 'Source_Service', 'Target_Service', 'syscall_Count',
               'Request_Intensity', 'Avg_Data_Sent', 'Avg_Data_Received', 'Error_Rate']

    df_pairwise = pd.DataFrame(pairwise_interactions, columns=columns)
    return df_pairwise

if __name__ == "__main__":
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    df = process_all_csv_in_folder(train_data)
    df = optimize_memory(df)
    # Assuming the DataFrame is named df, drop rows where send_syscall_count == 0
    df = df[df['total_operations'] != 0].reset_index(drop=True)
    df_test = process_all_csv_in_folder(test_data)
    df_test = optimize_memory(df_test)
    df_test = df_test[df_test['total_operations'] != 0].reset_index(drop=True)

    df_service= service_pairwise_data(df)
    # df_service = compute_trace_level_centrality(df_service)
    # print(df_service[["betweenness", "degree", "closeness"]].describe())
    df_service_test = service_pairwise_data(df_test)
    # df_service_test = compute_trace_level_centrality(df_service_test)
    # print(df_service_test[["betweenness", "degree", "closeness"]].describe())
    df_service.to_csv("df_service.csv", index=False)
    print(df_service.dtypes)
    print(df_service.head())

    # Apply heuristics
    # data = apply_heuristics(train_data)
    # file = sys.argv[1]
    # df_service= pd.read_csv(file)
    # Train models
    models = train_models(df_service, FEATURES)

    # Predict
    predictions = predict_clusters(df_service_test, models, FEATURES)

    # Combine heuristics and ML results
    data = pd.concat([df_service_test, predictions], axis=1)
    print(data.columns)
    data['Request_Intensity'] = data['Request_Intensity'].round(4)
    # data['betweenness'] = data['betweenness'].round(2)
    # data['degree'] = data['degree'].round(2)
    # data['closeness'] = data['closeness'].round(2)
    data.to_csv("data.csv", index=False)

    data['duplicate_count'] = \
    data.groupby(['Source_Service', 'Target_Service', ('Blob', 'MiniBatchKMeans'), ('Blob', 'GaussianMixture'),('Blob', 'KNNAnomalyDetection'),])[
        'Source_Service'].transform('count')

    filtered_df = data[
        ['Source_Service', 'Target_Service', ('Blob', 'MiniBatchKMeans'), ('Blob', 'GaussianMixture'),('Blob', 'KNNAnomalyDetection'),
          'duplicate_count']].drop_duplicates().reset_index(
        drop=True)

    filtered_df.to_csv("filtered_data.csv", index=False)
    visualize_anomaly_scores(models, predictions)