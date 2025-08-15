import sys
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------------
# CONFIGURATION
# -------------------------------
LABEL_MAP = {'Blob': 1, 'Normal': 0}

BLOB_FEATURES = {
    'Blob': ['syscall_Count',
             'Request_Intensity', 'Avg_Data_Sent', 'Avg_Data_Received', 'Error_Rate', 'span_duration']
    # 'Outlier': ['I', 'J', 'K']
}

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


# -------------------------------
# DATA PREPARATION
# -------------------------------
def prepare_data(df, feature_cols, label_col, label_map):
    X = StandardScaler().fit_transform(df[feature_cols])
    y = df[label_col].map(label_map).fillna(-1).astype(int)
    return X, y


# -------------------------------
# DATASET CLASS
# -------------------------------
class SemiDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


# -------------------------------
# SIMPLE MLP MODEL
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# TRAINING FUNCTION
# -------------------------------
def train_semi_supervised_model(X, y, epochs=20, batch_size=64):
    labeled_mask = y != -1
    dataset = SemiDataset(X[labeled_mask], y[labeled_mask])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP(X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


# -------------------------------
# INFERENCE FUNCTION
# -------------------------------
def infer(model, df, feature_cols):
    X = torch.tensor(StandardScaler().fit_transform(df[feature_cols]), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.numpy(), probs.numpy()

def save_model(model, path='saved_blob_model.pt'):
    torch.save(model.state_dict(), path)

def load_model(input_dim, path='saved_blob_model.pt'):
    model = MLP(input_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def service_pairwise_data(df):
    """
    Constructs a pairwise interaction dataset from span-level data and carries forward span_duration.
    Also calculates messaging features for weak labeling and model input.
    """

    df['time_stamp'] = pd.to_numeric(df['time_stamp'])
    df_sorted = df.sort_values(by=['trace_id', 'time_stamp'])

    pairwise_interactions = []

    for trace_id, group in df_sorted.groupby('trace_id'):
        services = group[
            ['service_name', 'recv_data', 'transmitted_data',
             'recv_syscall_count', 'send_syscall_count',
             'error_count', 'span_duration']].values

        for i in range(len(services) - 1):
            source, recv_data_src, transmitted_data_src, recv_sys_src, send_sys_src, error_src, duration_src = services[i]
            target, recv_data_tgt, transmitted_data_tgt, recv_sys_tgt, send_sys_tgt, error_tgt, duration_tgt = services[i + 1]

            syscall_count = recv_sys_src + send_sys_src + recv_sys_tgt + send_sys_tgt
            request_count = 1
            avg_response_time = (duration_src + duration_tgt) / 2
            request_intensity = request_count / avg_response_time if avg_response_time > 0 else 0

            avg_data_sent = (transmitted_data_src + transmitted_data_tgt) / 2
            avg_data_received = (recv_data_src + recv_data_tgt) / 2
            total_errors = error_src + error_tgt
            error_rate = total_errors / request_count

            # This is the approximate span duration assigned to this interaction (avg of two spans)
            span_duration = avg_response_time

            pairwise_interactions.append([
                group['time_stamp'].iloc[i], trace_id,
                source, target, syscall_count, request_intensity,
                avg_data_sent, avg_data_received, error_rate, span_duration
            ])

    columns = [
        'Timestamp', 'Trace_ID', 'Source_Service', 'Target_Service',
        'syscall_Count', 'Request_Intensity', 'Avg_Data_Sent',
        'Avg_Data_Received', 'Error_Rate', 'span_duration'
    ]

    df_pairwise = pd.DataFrame(pairwise_interactions, columns=columns)
    return df_pairwise

def detect_blob_component_exclusion_strategy_pairwise(df):
    """
    Applies the Component Exclusion Strategy to pairwise communication data.
    Labels Source_Service components as 'Blob' if their exclusion causes
    a statistically significant drop in total span_duration.
    """

    # Ensure span_duration exists or approximate it if needed
    if 'span_duration' not in df.columns:
        raise ValueError("Missing 'span_duration' column required for exclusion strategy.")

    # We use Source_Service as the component name
    df['service_name'] = df['Source_Service']

    # Filter only spans with meaningful messaging
    df_msg = df.copy()

    # Total messaging time
    omega_total = df_msg['span_duration'].sum()

    blob_flags = {}
    pi_zeta_values = {}
    for component in df_msg['service_name'].unique():
        df_excl = df_msg[df_msg['service_name'] != component]
        omega_excl = df_excl["span_duration"].sum()
        pi_zeta = 1 - (omega_excl / omega_total)
        pi_zeta_values[component] = pi_zeta

    # Convert to Series for easier handling
    pi_zeta_series = pd.Series(pi_zeta_values)

    # Compute z-scores
    z_scores = (pi_zeta_series - pi_zeta_series.mean()) / pi_zeta_series.std()

    # Choose a Z-score threshold (e.g., z > 1.0 means more than 1 std above mean)
    z_threshold = 0.7  # You can tune this (try 0.7–1.5 range)

    # Flag blobs based on Z-score
    blob_flags = (z_scores > z_threshold).to_dict()
    print("Z-score Threshold:", z_threshold)
    print(z_scores.sort_values(ascending=False).head(10))
    print("Mean pi_zeta:", pi_zeta_series.mean())
    print("Std pi_zeta:", pi_zeta_series.std())
    print("All pi_zeta values:\n", pi_zeta_series.sort_values(ascending=False))

    # Apply weak label
    df['weak_label'] = df['Source_Service'].map(lambda x: 'Blob' if blob_flags.get(x, False) else 'Normal')
    return df

def evaluation_vis(df_test):
    # ---------------------------
    # 1. CONFIDENCE DISTRIBUTION
    # ---------------------------
    plt.figure(figsize=(8, 4))
    sns.histplot(df_test['confidence'], bins=20, kde=True)
    plt.axvline(0.6, color='red', linestyle='--', label='Threshold = 0.6')
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("confidence_distribution.png", dpi=300)
    plt.show()

    # ---------------------------
    # 2. CONFUSION MATRIX (If you have true labels)
    # ---------------------------
    if 'true_label' in df_test.columns:
        from sklearn.metrics import ConfusionMatrixDisplay
        label_map = {0: 'Normal', 1: 'Blob'}
        y_true = df_test['true_label']
        y_pred = df_test['predicted_label']

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.values()))
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png", dpi=300)
        plt.show()

    # ---------------------------
    # 3. AGREEMENT WITH WEAK LABELS (if present)
    # ---------------------------
    if 'weak_label' in df_test.columns:
        weak_label_map = {'Normal': 0, 'Blob': 1}
        df_test['weak_label_numeric'] = df_test['weak_label'].map(weak_label_map)

        agreement = (df_test['predicted_label'] == df_test['weak_label_numeric']).mean()
        print(f"Agreement with weak labels: {agreement:.2%}")

        # Bar plot
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df_test, x='predicted_label',
                    y=(df_test['predicted_label'] == df_test['weak_label_numeric']).astype(int))
        plt.xticks([0, 1], ['Normal', 'Blob'])
        plt.ylim(0, 1)
        plt.ylabel("Agreement Rate")
        plt.title("Agreement with Weak Labels")
        plt.tight_layout()
        plt.savefig("agreement_with_weak_labels.png", dpi=300)
        plt.show()


def detect_blob_component_exclusion_strategy(df):
    """
    Implements the Component Exclusion Strategy for detecting Blob components
    and returns the original dataframe with weak labels added.
    """
    # Filter only messaging spans
    df_msg = df[df["total_operations"] > 0].copy()
    # df_msg = df
    # Total messaging time
    omega_total = df_msg["span_duration"].sum()

    blob_flags = {}
    # --- Compute pi_zeta for each component ---
    pi_zeta_values = {}
    for component in df_msg['service_name'].unique():
        df_excl = df_msg[df_msg['service_name'] != component]
        omega_excl = df_excl["span_duration"].sum()
        pi_zeta = 1 - (omega_excl / omega_total)
        pi_zeta_values[component] = pi_zeta

    # Convert to Series for easier handling
    pi_zeta_series = pd.Series(pi_zeta_values)

    # Compute z-scores
    z_scores = (pi_zeta_series - pi_zeta_series.mean()) / pi_zeta_series.std()

    # Choose a Z-score threshold (e.g., z > 1.0 means more than 1 std above mean)
    z_threshold = 1.5  # You can tune this (try 0.7–1.5 range)

    # Flag blobs based on Z-score
    blob_flags = (z_scores > z_threshold).to_dict()
    print("Z-score Threshold:", z_threshold)
    print(z_scores.sort_values(ascending=False).head(10))
    print("Mean pi_zeta:", pi_zeta_series.mean())
    print("Std pi_zeta:", pi_zeta_series.std())
    print("All pi_zeta values:\n", pi_zeta_series.sort_values(ascending=False))

    # Apply weak label
    df['weak_label'] = df['Source_Service'].map(lambda x: 'Blob' if blob_flags.get(x, False) else 'Normal')
    return df


if __name__ == "__main__":
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    df = process_all_csv_in_folder(train_data)
    df = optimize_memory(df)
    # Assuming the DataFrame is named df, drop rows where send_syscall_count == 0
    df = df[df['total_operations'] != 0].reset_index(drop=True)
    # df = detect_blob_component_exclusion_strategy(df)

    df_service= service_pairwise_data(df)
    df_train = df_service
    df_test_initial = process_all_csv_in_folder(test_data)
    df_test_initial = optimize_memory(df_test_initial)
    df_test_initial = df_test_initial[df_test_initial['total_operations'] != 0].reset_index(drop=True)
    # df_test_initial = detect_blob_component_exclusion_strategy(df_test_initial)
    df_service_test= service_pairwise_data(df_test_initial)
    df_test= df_service_test

    df_train = detect_blob_component_exclusion_strategy_pairwise(df_train)
    df_test = detect_blob_component_exclusion_strategy_pairwise(df_test)
    print(df_train['weak_label'].value_counts())

    # Example of loading and applying to sample file

    # Label training data with Mahalanobis
    # df_train['mahal_score'], df_train['weak_label'] = mahalanobis_weak_labeling(df_train, BLOB_FEATURES['Blob'])

    # Prepare training data
    X_train, y_train = prepare_data(df_train, BLOB_FEATURES['Blob'], 'weak_label', LABEL_MAP)

    # Train model
    model = train_semi_supervised_model(X_train, y_train)
    save_model(model, path='saved_blob_model.pt')

    # Inference on test data
    test_preds, test_probs = infer(model, df_test, BLOB_FEATURES['Blob'])
    df_test['predicted_label'] = test_preds
    df_test['confidence'] = test_probs.max(axis=1)
    # Flag low-confidence predictions
    CONFIDENCE_THRESHOLD = 0.6
    df_test['low_confidence'] = df_test['confidence'] < CONFIDENCE_THRESHOLD

    print("Sample Predictions:\n", df_test.head(20))
    df_test.to_csv("predicted_test_data.csv", index=False)
    print("\nSaved predictions to predicted_test_data.csv")

    # Ensure label column is readable
    df_test['predicted_label_name'] = df_test['predicted_label'].map({0: 'Normal', 1: 'Blob'})

    # Subsample for clarity if needed
    df_vis = df_test.sample(n=min(1000, len(df_test)), random_state=42)

    sns.pairplot(df_test, vars=BLOB_FEATURES['Blob'], hue='predicted_label_name', plot_kws={'alpha': 0.5})
    plt.suptitle("Pairplot of Blob Features Colored by Predicted Label", y=1.02)
    plt.tight_layout()
    plt.savefig("pairplot_predicted_labels.png", dpi=300)
    plt.show()

    for feature in BLOB_FEATURES['Blob']:
        sns.kdeplot(data=df_test, x=feature, hue='predicted_label_name')
        plt.title(f'Distribution of {feature} by Predicted Label')
        plt.savefig(f'distplot_{feature}.png')
        plt.clf()

    # weak_label_map = {'Normal': 0, 'Blob': 1}
    # df_test['weak_label_numeric'] = df_test['weak_label'].map(weak_label_map)
    # agreement = (df_test['predicted_label'] == df_test['weak_label_numeric']).mean()
    # print(f"Agreement with weak labels: {agreement:.2%}")

    evaluation_vis(df_test)

    # Step 1: Ensure we have readable labels
    df_test['predicted_label_name'] = df_test['predicted_label'].map({0: 'Normal', 1: 'Blob'})

    # Step 2: Aggregate to get final classification per pair
    CONFIDENCE_THRESHOLD = 0.6


    def classify_pair(group):
        if any((group['predicted_label'] == 1) & (group['confidence'] >= CONFIDENCE_THRESHOLD)):
            return 'Blob'
        else:
            return 'Normal'


    pair_results = (
        df_test
        .groupby(['Source_Service', 'Target_Service'])
        .apply(classify_pair)
        .reset_index()
        .rename(columns={0: 'final_classification'})
    )

    # Save and print results
    pair_results.to_csv("final_blob_predictions_per_pair.csv", index=False)
    print("\n✔️ Final Blob classification per service pair saved to 'final_blob_predictions_per_pair.csv'")
    print(pair_results.head())

    plt.figure(figsize=(8, 4))
    sns.countplot(data=pair_results, x='final_classification')
    plt.title("Final Classification Count per Service Pair")
    plt.xlabel("Predicted Class")
    plt.ylabel("Number of Pairs")
    plt.tight_layout()
    plt.savefig("final_blob_class_distribution.png", dpi=300)
    plt.show()









