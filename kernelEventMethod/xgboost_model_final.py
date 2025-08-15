import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# Function to preprocess the datasets for XGBoost
def data_preprocessing(df, event_op_name):
    df = df[df['event_op_name'] == event_op_name].copy()
    df = df.sort_values(by='cur_ts')

    X = df[['response_time', 'ret', 'count', 'event_error_count']]
    y = df['label']

    return X, y


# Function to train and evaluate the XGBoost model
def train_xgboost(X_train, X_test, y_train, y_test, event_op_name, results):
    xgb_model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=2,
    )

    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    # Evaluation
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True,
                                       target_names=['Baseline', 'Blob Anti-pattern', 'EST Anti-pattern'])
    conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)

    print(f'XGBoost Model Accuracy for {event_op_name}: {accuracy_xgb * 100:.2f}%')

    # Save results to the dictionary
    results.append({
        'event_op_name': event_op_name,
        'accuracy': accuracy_xgb,
        'classification_report': report_xgb,
        'confusion_matrix': conf_matrix_xgb.tolist()  # Convert to list for saving in a DataFrame
    })


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
    combined_df = combined_df.sample(frac=0.05, random_state=42).reset_index(drop=True)

    combined_df['response_time'] = pd.to_numeric(combined_df['response_time'], downcast='integer')
    combined_df['ret'] = pd.to_numeric(combined_df['ret'], downcast='integer')
    combined_df['count'] = pd.to_numeric(combined_df['count'], downcast='integer')
    combined_df['event_error_count'] = pd.to_numeric(combined_df['event_error_count'], downcast='integer')

    # Get unique event operation names
    event_op_names = combined_df['event_op_name'].unique()[:10]

    # Store results for each event operation name
    results = []

    # Loop through each event_op_name and train the model separately
    for event_op_name in event_op_names:
        print(f'Training model for {event_op_name}...')

        # Preprocess data
        X, y = data_preprocessing(combined_df, event_op_name)

        # Handle missing values
        X = X.fillna(X.mean())

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        # Train and evaluate XGBoost
        train_xgboost(X_train, X_test, y_train, y_test, event_op_name, results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV file
    results_df.to_csv('xgboost_event_op_results.csv', index=False)
    print('Results saved to xgboost_event_op_results.csv')


if __name__ == "__main__":
    main()
