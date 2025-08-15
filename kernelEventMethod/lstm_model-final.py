import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Function to preprocess the datasets for XGBoost
def data_preprocessing(df, label_num):
    sequences = []
    labels = []

    # Grouping by event operation name
    grouped = df.groupby('event_op_name')
    for name, group in grouped:
        group = group.sort_values(by='cur_ts')
        sequence = group[['response_time', 'ret', 'count', 'event_error_count']].values

        # Flattening the sequence (mean, std, min, max) as features
        features = np.concatenate([
            sequence.mean(axis=0),
            sequence.std(axis=0),
            sequence.min(axis=0),
            sequence.max(axis=0)
        ])

        sequences.append(features)
        print('features size')
        print(features.len())
        labels.append(label_num)

    sequences = np.array(sequences)
    labels = np.array(labels)
    return sequences, labels


# Function to create synthetic trace_id and prepare sequences for LSTM
def prepare_sequences_with_synthetic_id(df, feature_columns):
    # Create synthetic trace_id by grouping every N rows
    df['trace_id'] = (df.index // 10).astype(int)
    sequences = []
    labels = []
    grouped = df.groupby('trace_id')  # Group by synthetic trace_id
    for name, group in grouped:
        seq = group[feature_columns].values
        sequences.append(seq)
        labels.append(group['label'].iloc[0])  # Assuming all events in a sequence have the same label
    return sequences, labels


# Function to train and evaluate the XGBoost model
def train_xgboost(X_train, X_test, y_train, y_test):
    xgb_model = XGBClassifier(
        n_estimators=50,  # Reduced for lower resource usage
        max_depth=3,  # Reduced for lower resource usage
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=2  # Reduced for lower CPU usage
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    # Evaluation
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    report_xgb = classification_report(y_test, y_pred_xgb,
                                       target_names=['Baseline', 'Blob Anti-pattern', 'EST Anti-pattern'])
    conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)

    print(f'XGBoost Model Accuracy: {accuracy_xgb * 100:.2f}%')
    print('\nXGBoost Classification Report:\n', report_xgb)
    print('\nXGBoost Confusion Matrix:\n', conf_matrix_xgb)

    # Predict probabilities for each class
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)

    # Extracting the scores for visualization
    baseline_scores = y_pred_proba_xgb[:, 0]
    blob_scores = y_pred_proba_xgb[:, 1]
    est_scores = y_pred_proba_xgb[:, 2]

    # Print probability scores for each record
    for i, probs in enumerate(y_pred_proba_xgb):
        print(f'Record {i + 1}: blob_score={probs[1]:.2f}, baseline_score={probs[0]:.2f}, est_score={probs[2]:.2f}')

    # Evaluation
    y_pred_xgb = np.argmax(y_pred_proba_xgb, axis=1)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    report_xgb = classification_report(y_test, y_pred_xgb,
                                       target_names=['Baseline', 'Blob Anti-pattern', 'EST Anti-pattern'])
    conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
    visualization(conf_matrix_xgb, baseline_scores, blob_scores, est_scores)

    print(f'XGBoost Model Accuracy: {accuracy_xgb * 100:.2f}%')
    print('\nXGBoost Classification Report:\n', report_xgb)
    print('\nXGBoost Confusion Matrix:\n', conf_matrix_xgb)

def visualization(conf_matrix, baseline_scores, blob_scores, est_scores):
    # Define class names
    class_names = ['Baseline', 'Blob Anti-pattern', 'EST Anti-pattern']

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    # Save the figure
    plt.savefig('Confusion_Matrix.png', dpi=300, bbox_inches='tight')

    # To avoid any negative scores due to randomization, let's normalize the scores
    min_score = np.min([blob_scores, baseline_scores, est_scores], axis=0)
    blob_scores = blob_scores + np.abs(min_score)
    baseline_scores = baseline_scores + np.abs(min_score)
    est_scores = est_scores + np.abs(min_score)
    total_scores = blob_scores + baseline_scores + est_scores
    blob_scores /= total_scores
    baseline_scores /= total_scores
    est_scores /= total_scores

    # Scatter plot
    plt.figure(figsize=(10, 7))
    plt.scatter(blob_scores, baseline_scores, c=est_scores, cmap='viridis', s=100, alpha=0.6, edgecolors='w')
    plt.colorbar(label='EST Anti-pattern Score')
    plt.xlabel('Blob Anti-pattern Score')
    plt.ylabel('Baseline Score')
    plt.title('Scatter Plot of Classification Scores')
    plt.grid(True)
    plt.show()
    # Save the figure
    plt.savefig('anti_pattern_probabilities_scatter_plot.png', dpi=300, bbox_inches='tight')

    # 2D Histogram plot
    plt.figure(figsize=(10, 7))
    plt.hist2d(blob_scores, baseline_scores, bins=50, cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('Blob Anti-pattern Score')
    plt.ylabel('Baseline Score')
    plt.title('2D Histogram of Classification Scores')
    plt.grid(True)

    # Save the figure
    plt.savefig('2d_histogram_classification_scores.png', dpi=300, bbox_inches='tight')

    # Scatter plot with transparency
    plt.figure(figsize=(10, 7))
    plt.scatter(blob_scores, baseline_scores, c=est_scores, cmap='viridis', s=100, alpha=0.3, edgecolors='w')
    plt.colorbar(label='EST Anti-pattern Score')
    plt.xlabel('Blob Anti-pattern Score')
    plt.ylabel('Baseline Score')
    plt.title('Scatter Plot with Transparency to Show Density')
    plt.grid(True)

    # Save the figure
    plt.savefig('scatter_plot_transparency_classification_scores.png', dpi=300, bbox_inches='tight')

    # Hexbin plot with a different color map
    plt.figure(figsize=(10, 7))
    plt.hexbin(blob_scores, baseline_scores, gridsize=50, cmap='inferno', mincnt=1)
    plt.colorbar(label='Density')
    plt.xlabel('Blob Anti-pattern Score')
    plt.ylabel('Baseline Score')
    plt.title('Hexbin Plot of Classification Scores (Inferno Color Map)')
    plt.grid(True)

    # Save the figure
    plt.savefig('hexbin_plot_classification_scores_inferno.png', dpi=300, bbox_inches='tight')

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
    report_lstm = classification_report(y_true_lstm, y_pred_lstm,
                                        target_names=['Baseline', 'Blob Anti-pattern', 'EST Anti-pattern'])
    print('\nLSTM Classification Report:\n', report_lstm)


# Main function to orchestrate the workflow
def main():
    # Load the datasets
    baseline_not_agg_df = pd.read_csv("baseline_not_agg.txt")
    blob_not_agg_df = pd.read_csv("blob_not_agg.txt")
    est_not_agg_df = pd.read_csv("est_not_agg.txt")

    # Assign labels
    baseline_not_agg_df['label'] = 0
    blob_not_agg_df['label'] = 1
    est_not_agg_df['label'] = 2

    # Combine datasets
    combined_df = pd.concat([baseline_not_agg_df, blob_not_agg_df, est_not_agg_df], ignore_index=True)
    combined_df = combined_df.sample(frac=0.1, random_state=42).reset_index(drop=True)

    # Feature selection
    feature_columns = ['response_time', 'ret', 'count', 'event_error_count']
    X = combined_df[feature_columns]
    y = combined_df['label']

    # Handle missing values
    X = X.fillna(X.mean())

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Train and evaluate XGBoost
    train_xgboost(X_train, X_test, y_train, y_test)
    #
    # # Prepare sequences for LSTM using synthetic trace_id
    # baseline_sequences, baseline_labels = prepare_sequences_with_synthetic_id(baseline_not_agg_df, feature_columns)
    # blob_sequences, blob_labels = prepare_sequences_with_synthetic_id(blob_not_agg_df, feature_columns)
    # est_sequences, est_labels = prepare_sequences_with_synthetic_id(est_not_agg_df, feature_columns)
    #
    # # Combine sequences and labels
    # all_sequences = baseline_sequences + blob_sequences + est_sequences
    # all_labels = baseline_labels + blob_labels + est_labels
    #
    # # Pad sequences
    # max_seq_length = max(len(seq) for seq in all_sequences)
    # print('max_seq_length')
    # print(max_seq_length)
    # padded_sequences = pad_sequences(all_sequences, maxlen=max_seq_length, padding='post', dtype='float32')
    #
    # # Encode labels
    # label_encoder = LabelEncoder()
    # integer_labels = label_encoder.fit_transform(all_labels)
    # categorical_labels = to_categorical(integer_labels)

    # # Train-test split for LSTM
    # X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(padded_sequences, categorical_labels,
    #                                                                         test_size=0.2, random_state=42,
    #                                                                         stratify=integer_labels)
    #
    # # Train and evaluate LSTM
    # train_lstm(X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, max_seq_length, feature_columns)


# Entry point
if __name__ == "__main__":
    main()
