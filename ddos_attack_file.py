import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import threading
import queue

print("Starting Enhanced DDoS Detection System...")

def generate_sample_data(n_samples=100000):
    print(f"Generating sample data with {n_samples} samples...")
    np.random.seed(42)
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='s')
    normal = pd.DataFrame({
        'timestamp': timestamps[:int(n_samples * 0.9)],
        'packet_size': np.random.normal(500, 150, int(n_samples * 0.9)),
        'packet_rate': np.random.normal(50, 15, int(n_samples * 0.9)),
        'source_entropy': np.random.uniform(3, 5, int(n_samples * 0.9)),
        'destination_entropy': np.random.uniform(3, 5, int(n_samples * 0.9)),
        'flow_duration': np.random.exponential(scale=10, size=int(n_samples * 0.9)),
        'is_attack': 0
    })
    ddos = pd.DataFrame({
        'timestamp': timestamps[int(n_samples * 0.9):],
        'packet_size': np.random.normal(200, 50, n_samples - int(n_samples * 0.9)),
        'packet_rate': np.random.normal(500, 100, n_samples - int(n_samples * 0.9)),
        'source_entropy': np.random.uniform(0, 2, n_samples - int(n_samples * 0.9)),
        'destination_entropy': np.random.uniform(0, 1, n_samples - int(n_samples * 0.9)),
        'flow_duration': np.random.exponential(scale=2, size=n_samples - int(n_samples * 0.9)),
        'is_attack': 1
    })
    data = pd.concat([normal, ddos], ignore_index=True)
    data = data.sort_values('timestamp').reset_index(drop=True)
    print(f"Generated data shape: {data.shape}")
    return data

def feature_engineering(df):
    print("Performing feature engineering...")
    df['packet_inter_arrival_time'] = df['timestamp'].diff().dt.total_seconds()
    df['packet_size_ma'] = df['packet_size'].rolling(window=10).mean()
    df['packet_rate_std'] = df['packet_rate'].rolling(window=10).std()
    df['bytes_per_flow'] = df['packet_size'] * df['packet_rate']
    return df.dropna()

def preprocess_data(df):
    print("Preprocessing data...")
    X = df.drop(['is_attack', 'timestamp'], axis=1)
    y = df['is_attack']

    # Remove constant features
    variance_threshold = VarianceThreshold()
    X = pd.DataFrame(variance_threshold.fit_transform(X), columns=X.columns[variance_threshold.get_support()])

    # SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Feature selection
    selector = SelectKBest(f_classif, k='all')
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Get feature names after selection
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Selected features: {selected_features}")

    return X_train_selected, X_test_selected, y_train, y_test, selector, selected_features

def create_model_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    return pipeline

def train_model(X_train, y_train):
    print("Training model...")
    pipeline = create_model_pipeline()

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def interpret_model(model, X_test, y_test, feature_names):
    print("Interpreting model...")
    rf_classifier = model.named_steps['classifier']
    X_test_transformed = model.named_steps['scaler'].transform(X_test)

    # Permutation Importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importance_sorted = sorted(zip(perm_importance.importances_mean, feature_names), reverse=True)

    plt.figure(figsize=(10, 6))
    plt.bar([x[1] for x in perm_importance_sorted], [x[0] for x in perm_importance_sorted])
    plt.title("Feature Importance (Permutation Importance)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # SHAP values
    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = explainer.shap_values(X_test_transformed)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    shap.summary_plot(shap_values_to_plot, pd.DataFrame(X_test_transformed, columns=feature_names), plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()

def detect_anomalies(X):
    print("Detecting anomalies...")
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = isolation_forest.fit_predict(X)
    return anomalies

def simulate_real_time_detection(model, selector, data_queue, result_queue):
    while True:
        try:
            packet = data_queue.get(timeout=1)
            features = selector.transform(packet.reshape(1, -1))
            prediction = model.predict(features)
            result_queue.put(("DDoS Attack Detected" if prediction[0] == 1 else "Normal Traffic", packet))
        except queue.Empty:
            continue

# Flask app for API
app = Flask(__name__)

# Load your model and selector from saved files
model = joblib.load('ddos_detection_model.joblib')
selector = joblib.load('feature_selector.joblib')
selected_features = ['packet_size', 'packet_rate', 'source_entropy', 'destination_entropy', 'flow_duration',
                     'packet_inter_arrival_time', 'packet_size_ma', 'packet_rate_std', 'bytes_per_flow']

@app.route('/')
def home():
    return "DDoS Detection API is running. Use the /predict endpoint for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the JSON data from the request
            data = request.json
            # Ensure the JSON contains all required features
            features = np.array([data[feature] for feature in selected_features])
            # Reshape and apply the feature selector
            selected_features_transformed = selector.transform(features.reshape(1, -1))
            # Get the model prediction
            prediction = model.predict(selected_features_transformed)
            return jsonify({'prediction': 'DDoS Attack' if prediction[0] == 1 else 'Normal Traffic'})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Method Not Allowed'}), 405

if __name__ == "__main__":
    try:
        print("Main execution started...")

        # MLflow setup
        mlflow.set_experiment("DDoS Detection")

        with mlflow.start_run():
            data = generate_sample_data()
            data = feature_engineering(data)
            X_train, X_test, y_train, y_test, selector, selected_features = preprocess_data(data)

            model = train_model(X_train, y_train)
            evaluate_model(model, X_test, y_test)

            interpret_model(model, X_test, y_test, selected_features)

            anomalies = detect_anomalies(X_test)
            print(f"Detected {sum(anomalies == -1)} anomalies in test set")

            joblib.dump(model, 'ddos_detection_model.joblib')
            joblib.dump(selector, 'feature_selector.joblib')

        app.run(host='0.0.0.0', port=5000, debug=False)

    except Exception as e:
        print(f"An error occurred during main execution: {str(e)}")
