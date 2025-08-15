import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns


# Function to calculate normalized distance
def normalized_distance(measured, baseline):
    return 2 * (measured / (measured + baseline))


# Function to calculate the slope of linear regression
def calculate_slope(baseline, measured):
    if len(baseline) == 0 or len(measured) == 0:
        return np.array([0])  # Return 0 slope if data is empty or invalid
    reg = LinearRegression()
    baseline_reshaped = np.array(baseline).reshape(-1, 1)  # Reshaping for linear regression input
    reg.fit(baseline_reshaped, measured)
    slope = reg.coef_[0]
    return np.array([slope])  # Ensure it's always 1D


# Function to preprocess the datasets for XGBoost
def preprocess_xgboost(df, baseline_df, label_num):
    features = []
    labels = []

    for i, row in df.iterrows():
        baseline_row = baseline_df.iloc[i]

        # Normalized distance
        normalized_dist = normalized_distance(
            row[['response_time', 'ret', 'count', 'event_error_count']].values,
            baseline_row[['response_time', 'ret', 'count', 'event_error_count']].values
        )

        # Slope of linear regression
        slope = calculate_slope(
            baseline_row[['response_time', 'ret', 'count', 'event_error_count']].values,
            row[['response_time', 'ret', 'count', 'event_error_count']].values
        )

        # Ensure slope is a 1D array
        if slope.ndim == 0:
            slope = np.array([slope])

        # Aggregate the features
        feature_vector = np.concatenate([
            normalized_dist,  # 1D array of normalized distances
            slope.reshape(-1)  # Ensure slope is a 1D array
        ])

        features.append(feature_vector)
        labels.append(label_num)

    features = np.array(features)
    labels = np.array(labels)
    return features, labels


# Function to preprocess data for LSTM
def preprocess_lstm(df, baseline_df, label_num, feature_columns):
    sequences = []
    labels = []

    # Create synthetic trace_id by grouping every N rows
    df['trace_id'] = (df.index // 10).astype(int)
    baseline_df['trace_id'] = (baseline_df.index // 10).astype(int)

    grouped = df.groupby('trace_id')
    baseline_grouped = baseline_df.groupby('trace_id')

    for (name, group), (baseline_name, baseline_group) in zip(grouped, baseline_grouped):
        measured = group[feature_columns].values
        baseline = baseline_group[feature_columns].values

        # Normalized distances
        normalized_distances = normalized_distance(measured, baseline)
        slope = calculate_slope(baseline, measured)

        # Combine normalized distances and slope into sequence features
        sequence = np.hstack([normalized_distances, slope])
        sequences.append(sequence)
        labels.append(group['label'].iloc[0])  # Assuming all events in a sequence have the same label

    return sequences, labels


# Function to train and evaluate the XGBoost model
def train_xgboost(X_train, X_test, y_train, y_test):
    xgb_model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=2
    )
    xgb_model.fit(X_train, y_train)

    # Predict probabilities for each class
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)

    # Evaluation
    y_pred_xgb = np.argmax(y_pred_proba_xgb, axis=1)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    report_xgb = classification_report(y_test, y_pred_xgb, target_names=['Baseline', 'Blob Anti-pattern', 'EST Anti-pattern'])
    conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
    visualize_results(conf_matrix_xgb, y_pred_proba_xgb)

    print(f'XGBoost Model Accuracy: {accuracy_xgb * 100:.2f}%')
    print('\nXGBoost Classification Report:\n', report_xgb)
    print('\nXGBoost Confusion Matrix:\n', conf_matrix_xgb)


# Function to train and evaluate the LSTM model
def train_lstm(X_train, X_test, y_train, y_test, max_seq_length, feature_columns):
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(max_seq_length, len(feature_columns)), return_sequences=True))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(LSTM(32))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(3, activation='softmax'))  # 3 classes

    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the LSTM model
    history = lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the LSTM model
    loss_lstm, accuracy_lstm = lstm_model.evaluate(X_test, y_test)
    print(f'LSTM Model Accuracy: {accuracy_lstm * 100:.2f}%')

    # LSTM Prediction and Classification Report
    y_pred_probs_lstm = lstm_model.predict(X_test)
    y_pred_lstm = np.argmax(y_pred_probs_lstm, axis=1)
    y_true_lstm = np.argmax(y_test, axis=1)
    report_lstm = classification_report(y_true_lstm, y_pred_lstm, target_names=['Baseline', 'Blob Anti-pattern', 'EST Anti-pattern'])
    print('\nLSTM Classification Report:\n', report_lstm)


# Visualization for confusion matrix and score distribution
def visualize_results(conf_matrix, y_pred_proba):
    class_names = ['Baseline', 'Blob Anti-pattern', 'EST Anti-pattern']

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Extract the predicted scores
    baseline_scores = y_pred_proba[:, 0]
    blob_scores = y_pred_proba[:, 1]
    est_scores = y_pred_proba[:, 2]

    # Scatter plot for score distribution
    plt.figure(figsize=(10, 7))
    plt.scatter(blob_scores, baseline_scores, c=est_scores, cmap='viridis', s=100, alpha=0.6, edgecolors='w')
    plt.colorbar(label='EST Anti-pattern Score')
    plt.xlabel('Blob Anti-pattern Score')
    plt.ylabel('Baseline Score')
    plt.title('Scatter Plot of Classification Scores')
    plt.grid(True)
    plt.show()


# Main function to orchestrate the workflow
def main():
    # Load the datasets
    baseline_df = pd.read_csv("baseline_not_agg.txt")
    blob_df = pd.read_csv("blob_not_agg.txt")
    est_df = pd.read_csv("est_not_agg.txt")

    # Assign labels
    baseline_df['label'] = 0
    blob_df['label'] = 1
    est_df['label'] = 2

    # Combine datasets
    combined_df = pd.concat([blob_df, est_df], ignore_index=True).sample(frac=0.1, random_state=42).reset_index(drop=True)

    # Feature selection
    feature_columns = ['response_time', 'ret', 'count', 'event_error_count']

    # Preprocess for XGBoost
    xgb_blob_features, xgb_blob_labels = preprocess_xgboost(blob_df, baseline_df, 1)
    xgb_est_features, xgb_est_labels = preprocess_xgboost(est_df, baseline_df, 2)

    # Combine XGBoost features and labels
    X_xgb = np.concatenate([xgb_blob_features, xgb_est_features], axis=0)
    y_xgb = np.concatenate([xgb_blob_labels, xgb_est_labels], axis=0)

    # Train-test split for XGBoost
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)

    # Train and evaluate XGBoost model
    print("Training XGBoost...")
    train_xgboost(X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb)

    # # Train and evaluate LSTM
    # print("\nTraining and evaluating LSTM...")
    # train_evaluate_lstm(
    #     X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm,
    #     max_seq_length, feature_columns
    # )
    #

# Entry point
if __name__ == "__main__":
    main()
