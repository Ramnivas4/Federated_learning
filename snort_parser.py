import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_snort_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def preprocess_snort_data(df):
    # Drop timestamp column (not useful for numerical processing)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"], errors='ignore')
    
    # Define categorical and numerical columns
    categorical_columns = ["alert", "classification", "protocol", "src_ip", "dst_ip"]
    numerical_columns = ["priority", "src_port", "dst_port"]
    df_columns = df.columns.tolist()
    
    # Ensure required columns exist
    missing_columns = [col for col in categorical_columns + numerical_columns if col not in df_columns]
    if missing_columns:
        print(f"Warning: Missing columns in dataset: {missing_columns}. Processing only available columns.")
        categorical_columns = [col for col in categorical_columns if col in df_columns]
        numerical_columns = [col for col in numerical_columns if col in df_columns]
    
    # Encode categorical fields
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    
    # Normalize numerical fields
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return df, label_encoders

def train_autoencoder(data):
    input_dim = data.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(data, data, epochs=10, batch_size=32, verbose=1)
    
    return model

def detect_anomalies(model, data, threshold=0.05):
    reconstructions = model.predict(data)
    loss = np.mean(np.abs(reconstructions - data), axis=1)
    anomalies = loss > threshold
    return anomalies

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python snort_parser.py <dataset_file>")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    df = load_snort_data(dataset_file)
    processed_data, label_encoders = preprocess_snort_data(df)
    
    if processed_data.empty:
        print("Error: Processed data is empty. Check the input dataset.")
        sys.exit(1)
    
    autoencoder = train_autoencoder(processed_data)
    anomalies = detect_anomalies(autoencoder, processed_data)
    
    df["Anomaly"] = anomalies.astype(int)
    df_filtered = df[df["Anomaly"] == 0]
    
    if df_filtered.empty:
        print("Warning: All data points were detected as anomalies. Keeping some normal samples.")
        df_filtered = df.nsmallest(10, "Anomaly")  # Keep 10 least anomalous records
    
    # Ensure classification column is retained and encoded
    if "classification" in df.columns:
        df_filtered["classification"] = label_encoders["classification"].transform(df_filtered["classification"].astype(str))
    
    # Keep only numeric columns + classification
    df_filtered = df_filtered.select_dtypes(include=[np.number])
    
    df_filtered.to_csv("processed_snort_data.csv", index=False)
    print("Processed Snort data saved to 'processed_snort_data.csv'")
