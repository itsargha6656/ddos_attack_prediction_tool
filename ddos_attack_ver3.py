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
import logging


class DDoSDetectionSystem:
    def __init__(self):
        self.model = None
        self.selector = None
        self.scaler = None
        self.selected_features = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def generate_sample_data(self, n_samples=100000):
        self.logger.info(f"Generating sample data with {n_samples} samples...")
        np.random.seed(42)
        timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='s')

        # Normal traffic patterns
        normal = pd.DataFrame({
            'timestamp': timestamps[:int(n_samples * 0.9)],
            'packet_size': np.random.normal(500, 100, int(n_samples * 0.9)),
            'packet_rate': np.random.normal(100, 30, int(n_samples * 0.9)),
            'source_entropy': np.random.uniform(4, 5, int(n_samples * 0.9)),
            'destination_entropy': np.random.uniform(4, 5, int(n_samples * 0.9)),
            'flow_duration': np.random.exponential(scale=5, size=int(n_samples * 0.9)),
            'is_attack': 0
        })

        # DDoS attack patterns
        ddos = pd.DataFrame({
            'timestamp': timestamps[int(n_samples * 0.9):],
            'packet_size': np.random.normal(100, 20, n_samples - int(n_samples * 0.9)),
            'packet_rate': np.random.normal(3000, 1000, n_samples - int(n_samples * 0.9)),
            'source_entropy': np.random.uniform(0, 2, n_samples - int(n_samples * 0.9)),
            'destination_entropy': np.random.uniform(0, 1.5, n_samples - int(n_samples * 0.9)),
            'flow_duration': np.random.exponential(scale=0.1, size=n_samples - int(n_samples * 0.9)),
            'is_attack': 1
        })

        data = pd.concat([normal, ddos], ignore_index=True)
        data = data.sort_values('timestamp').reset_index(drop=True)
        return data

    def feature_engineering(self, df):
        self.logger.info("Performing feature engineering...")
        df['packet_inter_arrival_time'] = df['timestamp'].diff().dt.total_seconds()
        df['packet_size_ma'] = df['packet_size'].rolling(window=10).mean()
        df['packet_rate_std'] = df['packet_rate'].rolling(window=10).std()
        df['bytes_per_flow'] = df['packet_size'] * df['packet_rate']
        df['packets_per_second'] = df['packet_rate']
        df['bytes_per_second'] = df['bytes_per_flow'] / np.maximum(df['flow_duration'], 1e-6)
        return df.dropna()

    def train_model(self):
        self.logger.info("Starting model training process...")

        # Generate and process data
        data = self.generate_sample_data()
        data = self.feature_engineering(data)

        # Preprocess data
        X = data.drop(['is_attack', 'timestamp'], axis=1)
        y = data['is_attack']

        # Remove constant features
        variance_threshold = VarianceThreshold()
        X = pd.DataFrame(variance_threshold.fit_transform(X), columns=X.columns[variance_threshold.get_support()])

        # SMOTE for balancing
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Feature selection
        self.selector = SelectKBest(f_classif, k='all')
        X_train_selected = self.selector.fit_transform(X_train, y_train)
        X_test_selected = self.selector.transform(X_test)

        self.selected_features = X.columns[self.selector.get_support()].tolist()

        # Create and train model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                class_weight={0: 1, 1: 2},
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1
            ))
        ])

        pipeline.fit(X_train_selected, y_train)
        self.model = pipeline
        self.scaler = pipeline.named_steps['scaler']

        # Evaluate model
        y_pred = pipeline.predict(X_test_selected)
        y_prob = pipeline.predict_proba(X_test_selected)[:, 1]

        self.logger.info("\nModel Evaluation Results:")
        self.logger.info("\nConfusion Matrix:")
        self.logger.info(confusion_matrix(y_test, y_pred))
        self.logger.info("\nClassification Report:")
        self.logger.info(classification_report(y_test, y_pred))
        self.logger.info("\nROC AUC Score:")
        self.logger.info(roc_auc_score(y_test, y_prob))

        return self.model

    def domain_specific_rules(self, features):
        score = 0
        reasons = []

        # Rule 1: Packet rate analysis
        if features['packet_rate'] > 3000:
            score += 0.5
            reasons.append(f"Critical: Extremely high packet rate: {features['packet_rate']}")
        elif features['packet_rate'] > 1000:
            score += 0.3
            reasons.append(f"Warning: High packet rate: {features['packet_rate']}")

        # Rule 2: Entropy analysis
        if features['source_entropy'] < 2.0 and features['destination_entropy'] < 2.0:
            score += 0.4
            reasons.append(
                f"Critical: Very low entropy: source={features['source_entropy']}, dest={features['destination_entropy']}")
        elif features['source_entropy'] < 3.0 or features['destination_entropy'] < 3.0:
            score += 0.2
            reasons.append(
                f"Warning: Low entropy detected: source={features['source_entropy']}, dest={features['destination_entropy']}")

        # Additional rules
        if features['flow_duration'] < 0.1:
            score += 0.3
            reasons.append(f"Critical: Extremely short flow duration: {features['flow_duration']}")

        if features['bytes_per_flow'] > 1000000:
            score += 0.3
            reasons.append(f"Critical: Very high bytes per flow: {features['bytes_per_flow']}")

        return score, reasons

    def predict(self, input_data):
        try:
            if not self.model:
                raise ValueError("Model not trained. Please train the model first.")

            # Calculate packets_per_second if not provided
            if 'packets_per_second' not in input_data:
                input_data['packets_per_second'] = input_data.get('packet_rate', 0)

            # Validate input features
            missing_features = set(self.selected_features) - set(input_data.keys())
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            # Prepare features
            features = {feature: input_data[feature] for feature in self.selected_features}

            # Get rule-based score
            rule_score, reasons = self.domain_specific_rules(features)

            # Prepare features for model prediction
            features_array = np.array([features[feature] for feature in self.selected_features]).reshape(1, -1)
            selected_features_transformed = self.selector.transform(features_array)

            # Get model prediction
            prediction_proba = self.model.predict_proba(selected_features_transformed)[0, 1]

            # Combine scores
            final_score = max(prediction_proba, rule_score)
            is_attack = final_score > 0.3

            # Get feature importances
            feature_importances = self.model.named_steps['classifier'].feature_importances_
            importance_dict = dict(zip(self.selected_features, feature_importances.tolist()))

            return {
                'prediction': 'DDoS Attack' if is_attack else 'Normal Traffic',
                'probability': float(final_score),
                'model_probability': float(prediction_proba),
                'rule_score': float(rule_score),
                'reasons': reasons,
                'feature_importances': importance_dict,
                'alert_level': 'High' if final_score > 0.6 else 'Medium' if final_score > 0.3 else 'Low'
            }

        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

            def generate_sample_data(self, n_samples=100000):
        self.logger.info(f"Generating sample data with {n_samples} samples...")
        np.random.seed(42)
        timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='s')

        # Normal traffic patterns
        normal = pd.DataFrame({
            'timestamp': timestamps[:int(n_samples * 0.9)],
            'packet_size': np.random.normal(500, 100, int(n_samples * 0.9)),
            'packet_rate': np.random.normal(100, 30, int(n_samples * 0.9)),
            'source_entropy': np.random.uniform(4, 5, int(n_samples * 0.9)),
            'destination_entropy': np.random.uniform(4, 5, int(n_samples * 0.9)),
            'flow_duration': np.random.exponential(scale=5, size=int(n_samples * 0.9)),
            'is_attack': 0
        })

        # DDoS attack patterns
        ddos = pd.DataFrame({
            'timestamp': timestamps[int(n_samples * 0.9):],
            'packet_size': np.random.normal(100, 20, n_samples - int(n_samples * 0.9)),
            'packet_rate': np.random.normal(3000, 1000, n_samples - int(n_samples * 0.9)),
            'source_entropy': np.random.uniform(0, 2, n_samples - int(n_samples * 0.9)),
            'destination_entropy': np.random.uniform(0, 1.5, n_samples - int(n_samples * 0.9)),
            'flow_duration': np.random.exponential(scale=0.1, size=n_samples - int(n_samples * 0.9)),
            'is_attack': 1
        })

        data = pd.concat([normal, ddos], ignore_index=True)
        data = data.sort_values('timestamp').reset_index(drop=True)
        return data

        def feature_engineering(self, df):
            self.logger.info("Performing feature engineering...")

        df['packet_inter_arrival_time'] = df['timestamp'].diff().dt.total_seconds()
        df['packet_size_ma'] = df['packet_size'].rolling(window=10).mean()
        df['packet_rate_std'] = df['packet_rate'].rolling(window=10).std()
        df['bytes_per_flow'] = df['packet_size'] * df['packet_rate']
        df['packets_per_second'] = df['packet_rate']
        df['bytes_per_second'] = df['bytes_per_flow'] / np.maximum(df['flow_duration'], 1e-6)
        return df.dropna()

        def train_model(self):
            self.logger.info("Starting model training process...")

        # Generate and process data
        data = self.generate_sample_data()
        data = self.feature_engineering(data)

        # Preprocess data
        X = data.drop(['is_attack', 'timestamp'], axis=1)
        y = data['is_attack']

        # Remove constant features
        variance_threshold = VarianceThreshold()
        X = pd.DataFrame(variance_threshold.fit_transform(X), columns=X.columns[variance_threshold.get_support()])

        # SMOTE for balancing
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Feature selection
        self.selector = SelectKBest(f_classif, k='all')
        X_train_selected = self.selector.fit_transform(X_train, y_train)
        X_test_selected = self.selector.transform(X_test)

        self.selected_features = X.columns[self.selector.get_support()].tolist()

        # Create and train model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                class_weight={0: 1, 1: 2},
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1
            ))
        ])

        pipeline.fit(X_train_selected, y_train)
        self.model = pipeline
        self.scaler = pipeline.named_steps['scaler']

        # Evaluate model
        y_pred = pipeline.predict(X_test_selected)
        y_prob = pipeline.predict_proba(X_test_selected)[:, 1]

        self.logger.info("\nModel Evaluation Results:")
        self.logger.info("\nConfusion Matrix:")
        self.logger.info(confusion_matrix(y_test, y_pred))
        self.logger.info("\nClassification Report:")
        self.logger.info(classification_report(y_test, y_pred))
        self.logger.info("\nROC AUC Score:")
        self.logger.info(roc_auc_score(y_test, y_prob))

        return self.model

        def domain_specific_rules(self, features):
            score = 0

        reasons = []

        # Rule 1: Packet rate analysis
        if features['packet_rate'] > 3000:
            score += 0.5
            reasons.append(f"Critical: Extremely high packet rate: {features['packet_rate']}")
        elif features['packet_rate'] > 1000:
            score += 0.3
            reasons.append(f"Warning: High packet rate: {features['packet_rate']}")

        # Rule 2: Entropy analysis
        if features['source_entropy'] < 2.0 and features['destination_entropy'] < 2.0:
            score += 0.4
            reasons.append(
                f"Critical: Very low entropy: source={features['source_entropy']}, dest={features['destination_entropy']}")
        elif features['source_entropy'] < 3.0 or features['destination_entropy'] < 3.0:
            score += 0.2
            reasons.append(
                f"Warning: Low entropy detected: source={features['source_entropy']}, dest={features['destination_entropy']}")

        # Additional rules
        if features['flow_duration'] < 0.1:
            score += 0.3
            reasons.append(f"Critical: Extremely short flow duration: {features['flow_duration']}")

        if features['bytes_per_flow'] > 1000000:
            score += 0.3
            reasons.append(f"Critical: Very high bytes per flow: {features['bytes_per_flow']}")

        return score, reasons

        def domain_specific_rules(self, features):
            score = 0

        reasons = []

        # Rule 1: Packet rate analysis
        if features['packet_rate'] > 3000:
            score += 0.5
            reasons.append(f"Critical: Extremely high packet rate: {features['packet_rate']}")
        elif features['packet_rate'] > 1000:
            score += 0.3
            reasons.append(f"Warning: High packet rate: {features['packet_rate']}")

        # Rule 2: Entropy analysis
        if features['source_entropy'] < 2.0 and features['destination_entropy'] < 2.0:
            score += 0.4
            reasons.append(
                f"Critical: Very low entropy: source={features['source_entropy']}, dest={features['destination_entropy']}")
        elif features['source_entropy'] < 3.0 or features['destination_entropy'] < 3.0:
            score += 0.2
            reasons.append(
                f"Warning: Low entropy detected: source={features['source_entropy']}, dest={features['destination_entropy']}")

        # Additional rules
        if features['flow_duration'] < 0.1:
            score += 0.3
            reasons.append(f"Critical: Extremely short flow duration: {features['flow_duration']}")

        if features['bytes_per_flow'] > 1000000:
            score += 0.3
            reasons.append(f"Critical: Very high bytes per flow: {features['bytes_per_flow']}")

        return score, reasons

        def predict(self, input_data):
            try:
                if not self.model:
                    raise ValueError("Model not trained. Please train the model first.")

                # Calculate packets_per_second if not provided
                if 'packets_per_second' not in input_data:
                    input_data['packets_per_second'] = input_data.get('packet_rate', 0)

                # Validate input features
                missing_features = set(self.selected_features) - set(input_data.keys())
                if missing_features:
                    raise ValueError(f"Missing required features: {missing_features}")

                # Prepare features
                features = {feature: input_data[feature] for feature in self.selected_features}

                # Get rule-based score
                rule_score, reasons = self.domain_specific_rules(features)

                # Prepare features for model prediction
                features_array = np.array([features[feature] for feature in self.selected_features]).reshape(1, -1)
                selected_features_transformed = self.selector.transform(features_array)

                # Get model prediction
                prediction_proba = self.model.predict_proba(selected_features_transformed)[0, 1]

                # Combine scores
                final_score = max(prediction_proba, rule_score)
                is_attack = final_score > 0.3

                # Get feature importances
                feature_importances = self.model.named_steps['classifier'].feature_importances_
                importance_dict = dict(zip(self.selected_features, feature_importances.tolist()))

                return {
                    'prediction': 'DDoS Attack' if is_attack else 'Normal Traffic',
                    'probability': float(final_score),
                    'model_probability': float(prediction_proba),
                    'rule_score': float(rule_score),
                    'reasons': reasons,
                    'feature_importances': importance_dict,
                    'alert_level': 'High' if final_score > 0.6 else 'Medium' if final_score > 0.3 else 'Low'
                }

            except Exception as e:
                self.logger.error(f"Error during prediction: {str(e)}")
                raise

                def detect_anomalies(X):

    print("Detecting anomalies...")
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = isolation_forest.fit_predict(X)
    return anomalies
    # Create an instance of the detection system


detector = DDoSDetectionSystem()

# Train the model
detector.train_model()

# Test with a normal traffic sample
normal_traffic = {
    'packet_size': 500,
    'packet_rate': 100,
    'source_entropy': 4.5,
    'destination_entropy': 4.5,
    'flow_duration': 5.0,
    'bytes_per_flow': 50000,
    'bytes_per_second': 10000,
    'packets_per_second': 100,
    'packet_inter_arrival_time': 0.01,
    'packet_size_ma': 500,
    'packet_rate_std': 10
}

# Test with a DDoS attack sample
ddos_traffic = {
    'packet_size': 100,
    'packet_rate': 3500,
    'source_entropy': 1.5,
    'destination_entropy': 1.0,
    'flow_duration': 0.05,
    'bytes_per_flow': 1500000,
    'bytes_per_second': 6000000,
    'packets_per_second': 3500,
    'packet_inter_arrival_time': 0.0003,
    'packet_size_ma': 100,
    'packet_rate_std': 500
}

# Make predictions
normal_result = detector.predict(normal_traffic)
ddos_result = detector.predict(ddos_traffic)

print("\nNormal Traffic Detection Result:")
print(normal_result)
print("\nDDoS Traffic Detection Result:")
print(ddos_result)
