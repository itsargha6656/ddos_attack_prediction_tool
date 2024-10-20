import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Add debug print statements
print("Starting DDoS detection script...")


def generate_sample_data(n_samples=10000):
    print(f"Generating sample data with {n_samples} samples...")
    np.random.seed(42)

    normal = pd.DataFrame({
        'packet_size': np.random.normal(500, 150, n_samples),
        'packet_rate': np.random.normal(50, 15, n_samples),
        'source_entropy': np.random.uniform(3, 5, n_samples),
        'destination_entropy': np.random.uniform(3, 5, n_samples),
        'is_attack': 0
    })

    ddos = pd.DataFrame({
        'packet_size': np.random.normal(200, 50, n_samples // 10),
        'packet_rate': np.random.normal(500, 100, n_samples // 10),
        'source_entropy': np.random.uniform(0, 2, n_samples // 10),
        'destination_entropy': np.random.uniform(0, 1, n_samples // 10),
        'is_attack': 1
    })

    data = pd.concat([normal, ddos], ignore_index=True)
    print(f"Generated data shape: {data.shape}")
    return data


def preprocess_data(df):
    print("Preprocessing data...")
    df = df.sample(frac=1).reset_index(drop=True)

    X = df.drop('is_attack', axis=1)
    y = df['is_attack']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Preprocessed data shapes: X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train):
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model


def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def detect_ddos(model, scaler, packet_size, packet_rate, source_entropy, destination_entropy):
    features = np.array([[packet_size, packet_rate, source_entropy, destination_entropy]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    return "DDoS Attack Detected" if prediction[0] == 1 else "Normal Traffic"


if __name__ == "__main__":
    try:
        print("Main execution started...")
        data = generate_sample_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

        model = train_model(X_train, y_train)

        evaluate_model(model, X_test, y_test)

        print("\nSimulating real-time detection:")
        print(detect_ddos(model, scaler, packet_size=200, packet_rate=450, source_entropy=1.5, destination_entropy=0.5))
        print(detect_ddos(model, scaler, packet_size=500, packet_rate=50, source_entropy=4, destination_entropy=4))

        print("Script execution completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")