import tensorflow as tf
import flwr as fl
import pandas as pd
import numpy as np
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys

# Step 1: Run Snort Parser to generate processed data
dataset_file = "simulated_traffic.csv"  # Change this to your dataset
subprocess.run(["python", "snort_parser.py", dataset_file])

# Step 2: Load the newly processed Snort data
df = pd.read_csv("processed_snort_data.csv")

# Ensure all columns are numeric
df = df.select_dtypes(include=[np.number])

# Identify target variable (classification) if exists
if "classification" in df.columns:
    label_encoder = LabelEncoder()
    df["classification"] = label_encoder.fit_transform(df["classification"])
    num_classes = len(label_encoder.classes_)
else:
    print("Error: 'classification' column not found in dataset.")
    sys.exit(1)

# Split features and target
X = df.drop(columns=["classification"])
y = df["classification"]

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a classification model with correct output shape
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")  # Adjust output neurons based on unique labels
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Define Flower Client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        return loss, len(x_test), {"accuracy": accuracy}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
