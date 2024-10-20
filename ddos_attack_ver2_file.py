import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Starting Enhanced DDoS Detection System...")


def generate_sample_data(n_samples=100000):
    print(f"Generating sample data with {n_samples} samples...")
    np.random.seed(42)
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='s')

    # Modified normal traffic patterns
    normal = pd.DataFrame({
        'timestamp': timestamps[:int(n_samples * 0.9)],
        'packet_size': np.random.normal(500, 100, int(n_samples * 0.9)),
        'packet_rate': np.random.normal(100, 30, int(n_samples * 0.9)),  # Reduced normal packet rate
        'source_entropy': np.random.uniform(4, 5, int(n_samples * 0.9)),  # Increased normal entropy
        'destination_entropy': np.random.uniform(4, 5, int(n_samples * 0.9)),
        'flow_duration': np.random.exponential(scale=5, size=int(n_samples * 0.9)),
        'is_attack': 0
    })

    # Modified DDoS attack patterns
    ddos = pd.DataFrame({
        'timestamp': timestamps[int(n_samples * 0.9):],
        'packet_size': np.random.normal(100, 20, n_samples - int(n_samples * 0.9)),
        'packet_rate': np.random.normal(3000, 1000, n_samples - int(n_samples * 0.9)),  # Lower threshold for attacks
        'source_entropy': np.random.uniform(0, 2, n_samples - int(n_samples * 0.9)),  # Adjusted entropy range
        'destination_entropy': np.random.uniform(0, 1.5, n_samples - int(n_samples * 0.9)),
        'flow_duration': np.random.exponential(scale=0.1, size=n_samples - int(n_samples * 0.9)),
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
    df['packets_per_second'] = df['packet_rate']  # Simplified calculation
    df['bytes_per_second'] = df['bytes_per_flow'] / np.maximum(df['flow_duration'], 1e-6)
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

    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Selected features: {selected_features}")

    return X_train_selected, X_test_selected, y_train, y_test, selector, selected_features

def create_model_pipeline():
    # Modified RandomForestClassifier parameters for better attack detection
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=300,          # Increased from 200
            max_depth=15,              # Adjusted for better generalization
            class_weight={0: 1, 1: 2}, # Give more weight to attack class
            random_state=42,
            min_samples_split=5,       # Added to prevent overfitting
            min_samples_leaf=2,        # Added to prevent overfitting
            n_jobs=-1                  # Use all available cores
        ))
    ])
    return pipeline

def train_model(X_train, y_train):
    print("Training model...")
    pipeline = create_model_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nROC AUC Score:")
    print(roc_auc_score(y_test, y_prob))

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

    print("Feature Importance:")
    for importance, feature in perm_importance_sorted:
        print(f"{feature}: {importance}")

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

# def calculate_packets_per_second(data):
#     """Calculate packets per second from packet rate and inter-arrival time"""
#     if data.get('packet_inter_arrival_time', 0) <= 0:
#         return data.get('packet_rate', 0)
#     return data.get('packet_rate', 0)


def domain_specific_rules(features):
    score = 0
    reasons = []

    # Modified thresholds and scoring for better attack detection

    # Rule 1: Packet rate analysis (Most important indicator)
    if features['packet_rate'] > 3000:  # Lowered threshold
        score += 0.5
        reasons.append(f"Critical: Extremely high packet rate: {features['packet_rate']}")
    elif features['packet_rate'] > 1000:
        score += 0.3
        reasons.append(f"Warning: High packet rate: {features['packet_rate']}")

    # Rule 2: Entropy analysis (Strong indicator of DDoS)
    if features['source_entropy'] < 2.0 and features['destination_entropy'] < 2.0:
        score += 0.4
        reasons.append(
            f"Critical: Very low entropy: source={features['source_entropy']}, dest={features['destination_entropy']}")
    elif features['source_entropy'] < 3.0 or features['destination_entropy'] < 3.0:
        score += 0.2
        reasons.append(
            f"Warning: Low entropy detected: source={features['source_entropy']}, dest={features['destination_entropy']}")

    # Rule 3: Flow duration analysis
    if features['flow_duration'] < 0.1:
        score += 0.3
        reasons.append(f"Critical: Extremely short flow duration: {features['flow_duration']}")
    elif features['flow_duration'] < 0.5:
        score += 0.2
        reasons.append(f"Warning: Short flow duration: {features['flow_duration']}")

    # Rule 4: Bytes per flow analysis
    if features['bytes_per_flow'] > 1000000:
        score += 0.3
        reasons.append(f"Critical: Very high bytes per flow: {features['bytes_per_flow']}")
    elif features['bytes_per_flow'] > 500000:
        score += 0.2
        reasons.append(f"Warning: High bytes per flow: {features['bytes_per_flow']}")

    # Rule 5: Packets per second analysis
    if features['packets_per_second'] > 3000:
        score += 0.4
        reasons.append(f"Critical: Very high packets per second: {features['packets_per_second']}")
    elif features['packets_per_second'] > 1000:
        score += 0.2
        reasons.append(f"Warning: High packets per second: {features['packets_per_second']}")

    # Rule 6: Bytes per second analysis
    if features['bytes_per_second'] > 5000000:
        score += 0.3
        reasons.append(f"Critical: Very high bytes per second: {features['bytes_per_second']}")
    elif features['bytes_per_second'] > 1000000:
        score += 0.2
        reasons.append(f"Warning: High bytes per second: {features['bytes_per_second']}")

    return score, reasons

def calculate_packets_per_second(data):
    """Calculate packets per second from packet rate"""
    return data.get('packet_rate', 0)  # Simplified calculation



app = Flask(__name__)

# Global variables for model and preprocessing components
model = None
selector = None
scaler = None
selected_features = None

@app.route('/')
def home():
    return "DDoS Detection API is running. Use the /predict endpoint for predictions."


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.json
            logger.info(f"Received input: {data}")

            # Calculate packets_per_second if not provided
            if 'packets_per_second' not in data:
                data['packets_per_second'] = calculate_packets_per_second(data)

            # Ensure all required features are present
            required_features = set(selected_features)
            received_features = set(data.keys())
            missing_features = required_features - received_features

            if missing_features:
                return jsonify({
                    'error': f"Missing required features: {', '.join(missing_features)}",
                    'required_features': list(required_features),
                    'received_features': list(received_features)
                }), 400

            features = {feature: data[feature] for feature in selected_features}
            logger.info(f"Input features: {features}")

            # Get rule-based score
            rule_score, reasons = domain_specific_rules(features)
            logger.info(f"Rule-based score: {rule_score}")
            logger.info(f"Reasons for score: {reasons}")

            # Prepare features for model prediction
            features_array = np.array([features[feature] for feature in selected_features]).reshape(1, -1)
            selected_features_transformed = selector.transform(features_array)
            scaled_features = scaler.transform(selected_features_transformed)

            # Get model prediction
            prediction_proba = model.predict_proba(scaled_features)[0, 1]
            logger.info(f"Model prediction probability: {prediction_proba}")

            # Combine scores with more weight on rule-based system
            final_score = max(prediction_proba, rule_score)

            # Modified threshold for better attack detection
            is_attack = final_score > 0.3  # Lowered threshold from 0.5 to 0.3

            # Get feature importances
            feature_importances = model.named_steps['classifier'].feature_importances_
            importance_dict = dict(zip(selected_features, feature_importances.tolist()))

            return jsonify({
                'prediction': 'DDoS Attack' if is_attack else 'Normal Traffic',
                'probability': float(final_score),
                'model_probability': float(prediction_proba),
                'rule_score': float(rule_score),
                'reasons': reasons,
                'feature_importances': importance_dict,
                'alert_level': 'High' if final_score > 0.6 else 'Medium' if final_score > 0.3 else 'Low'
            })

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Method Not Allowed'}), 405


if __name__ == "__main__":
    try:
        print("Main execution started...")

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
            joblib.dump(model.named_steps['scaler'], 'scaler.joblib')
            joblib.dump(selected_features, 'selected_features.joblib')

        model = joblib.load('ddos_detection_model.joblib')
        selector = joblib.load('feature_selector.joblib')
        scaler = joblib.load('scaler.joblib')
        selected_features = joblib.load('selected_features.joblib')

        print("Model and components loaded successfully.")
        print(f"Selected features: {selected_features}")

        app.run(host='0.0.0.0', port=5000, debug=False)

    except Exception as e:
        print(f"An error occurred during main execution: {str(e)}")