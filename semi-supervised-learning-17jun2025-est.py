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
LABEL_MAP_EST = {'EST': 1, 'Normal': 0}

EST_FEATURES = {
    'EST': ['transmitted_data', 'send_syscall_count', 'avg_size_send_msg', 'error_count']
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
# WEAK LABELING - MAHALANOBIS
# -------------------------------
def mahalanobis_weak_labeling(df, feature_cols, threshold_percentile=95):
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cov_model = EmpiricalCovariance().fit(X_scaled)
    mean = cov_model.location_
    inv_cov = cov_model.precision_

    scores = np.array([mahalanobis(x, mean, inv_cov) for x in X_scaled])
    threshold = np.percentile(scores, threshold_percentile)
    weak_labels = np.where(scores > threshold, 'Blob', 'Normal')

    return scores, weak_labels


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
        self.y = torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.long)  # ✅ FIX HERE

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

def evaluation_vis(df_test):
    # ---------------------------
    # 1. CONFIDENCE DISTRIBUTION
    # ---------------------------
    plt.figure(figsize=(8, 4))
    sns.histplot(df_test['confidence_est'], bins=20, kde=True)
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


if __name__ == "__main__":
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    df = process_all_csv_in_folder(train_data)
    df = optimize_memory(df)
    # Assuming the DataFrame is named df, drop rows where send_syscall_count == 0
    df = df[df['total_operations'] != 0].reset_index(drop=True)
    df_train = df
    df_test_initial = process_all_csv_in_folder(test_data)
    df_test_initial = optimize_memory(df_test_initial)
    df_test_initial = df_test_initial[df_test_initial['total_operations'] != 0].reset_index(drop=True)
    df_test= df_test_initial

    df_train['mahal_score_est'], df_train['weak_label_est'] = mahalanobis_weak_labeling(df_train, EST_FEATURES['EST'])
    X_train_est, y_train_est = prepare_data(df_train, EST_FEATURES['EST'], 'weak_label_est', LABEL_MAP_EST)
    # Train model
    model_est = train_semi_supervised_model(X_train_est, y_train_est)
    save_model(model_est, path='saved_est_model.pt')

    CONFIDENCE_THRESHOLD = 0.6
    test_preds_est, test_probs_est = infer(model_est, df_test, EST_FEATURES['EST'])
    df_test['predicted_label_est'] = test_preds_est
    df_test['confidence_est'] = test_probs_est.max(axis=1)
    df_test['low_confidence_est'] = df_test['confidence_est'] < CONFIDENCE_THRESHOLD


    def classify_est_pair(group):
        if any((group['predicted_label_est'] == 1) & (group['confidence_est'] >= CONFIDENCE_THRESHOLD)):
            return 'EST'
        else:
            return 'Normal'


    df_test['final_classification_est'] = df_test.apply(
        lambda row: 'EST' if (
                    row['predicted_label_est'] == 1 and row['confidence_est'] >= CONFIDENCE_THRESHOLD) else 'Normal',
        axis=1
    )

    df_test.to_csv("final_classification_est.csv", index=False)
    print("\n✔️ Final EST classification per service pair saved to 'final_est_predictions_per_pair.csv'")

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df_test, x='final_classification_est')
    plt.title("Final Classification Count- EST")
    plt.xlabel("Predicted Class")
    plt.ylabel("Number of Pairs")
    plt.tight_layout()
    plt.savefig("final_est_class_distribution.png", dpi=300)
    plt.show()

    # Ensure label column is readable
    df_test['predicted_label_name'] = df_test['predicted_label_est'].map({0: 'Normal', 1: 'EST'})

    evaluation_vis(df_test)
    sns.pairplot(df_test, vars=EST_FEATURES['EST'], hue='predicted_label_name', plot_kws={'alpha': 0.5})
    plt.suptitle("Pairplot of Est Features Colored by Predicted Label", y=1.02)
    plt.tight_layout()
    plt.savefig("pairplot_predicted_labels.png", dpi=300)
    plt.show()








