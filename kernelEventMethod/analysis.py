import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

def plot(merged_df):
    # Plotting histograms for normalized distances for a quick overview
    metrics_to_plot = [
        'normalized_distance_blob_max_response_time',
        'normalized_distance_est_max_response_time',
        'normalized_distance_blob_mean_response_time',
        'normalized_distance_est_mean_response_time',
        'normalized_distance_blob_std_response_time',
        'normalized_distance_est_std_response_time'
    ]

    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i + 1)
        sns.histplot(merged_df[metric], kde=True)
        plt.title(metric)

    plt.tight_layout()
    plt.show()

def correlation_analysis(merged_df):
    corr_matrix = merged_df[corr_metrics].corr()

    # Plotting the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Selected Metrics')
    plt.show()

    #     High Correlation: Some metrics, such as normalized_distance_blob_max_response_time and normalized_distance_blob_mean_response_time, show a strong positive correlation. This suggests that these metrics might behave similarly and could be redundant if used together in a model.
    # Negative Correlation: Certain slope metrics have negative correlations with the normalized distance metrics, indicating that as the effect of an anti-pattern becomes more pronounced, the trend captured by the slope might change direction.
    # Low Correlation: Some metrics have low or no correlation, suggesting they capture different aspects of the data, making them potentially valuable as complementary features in a model.

def trend_analysis(merged_df):
    # Plotting the trends of selected metrics over time (cur_ts_numeric)
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(corr_metrics[:6]):
        plt.subplot(2, 3, i + 1)
        sns.lineplot(x=merged_df['cur_ts_numeric'], y=merged_df[metric])
        plt.title(f'Trend of {metric} over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel(metric)

    plt.tight_layout()
    plt.show()
    # The trend analysis over time (cur_ts_numeric) reveals the following:
    # Observations:
    # Gradual Changes: Some metrics show a gradual increase or decrease over time, which could indicate the presence of an anti-pattern as the system('s performance degrades.'
    #                                                                                                                                                 'Sharp Variations: In some cases, there are sharp variations that might correspond to specific events or load conditions that trigger the anti-patterns.)
    # Consistency: Certain metrics exhibit more consistent trends across time, potentially making them reliable indicators for detecting anti-patterns early.


def feature_selection(merged_df):
    # Define the target variable
    # We'll classify traces as either "baseline" (0) or "anti-pattern" (1 for blob, 2 for est)
    # Here, I'm assuming that the presence of significant normalized distances indicates an anti-pattern
    merged_df['target'] = 0
    merged_df.loc[(merged_df['normalized_distance_blob_max_response_time'] > 0.5) |
                  (merged_df['normalized_distance_blob_mean_response_time'] > 0.5), 'target'] = 1
    merged_df.loc[(merged_df['normalized_distance_est_max_response_time'] > 0.5) |
                  (merged_df['normalized_distance_est_mean_response_time'] > 0.5), 'target'] = 2

    # Select features for the model
    features = corr_metrics

    return features

def model_training(merged_df, features):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(merged_df[features], merged_df['target'], test_size=0.3,
                                                        random_state=42)
    # Train a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Impute missing values using the median
    imputer = SimpleImputer(strategy='median')

    # Impute the training and testing data
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train the RandomForestClassifier on the imputed data
    clf.fit(X_train_imputed, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test_imputed)

    # Evaluate the model
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf

def feature_importance(clf):
    # Extract feature importances from the trained model
    importances = clf.feature_importances_

    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print(feature_importance_df)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in RandomForest Model')
    plt.gca().invert_yaxis()  # To have the most important feature on top
    plt.show()


if __name__ == "__main__":
    # Load the dataset
    merged_df = pd.read_csv(sys.argv[1])
    # Compute the correlation matrix for selected metrics
    corr_metrics = [
        'normalized_distance_blob_max_response_time',
        'normalized_distance_est_max_response_time',
        'normalized_distance_blob_mean_response_time',
        'normalized_distance_est_mean_response_time',
        'normalized_distance_blob_std_response_time',
        'normalized_distance_est_std_response_time',
        'slope_est_max_response_time',
        'slope_est_mean_response_time',
        'slope_est_std_response_time'
    ]
    plot(merged_df)
    correlation_analysis(merged_df)
    features= feature_selection(merged_df)
    trend_analysis(merged_df)
    clf= model_training(merged_df, features)
    feature_importance(clf)

