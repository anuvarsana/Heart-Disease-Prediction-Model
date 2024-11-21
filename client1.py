import pandas as pd
import socket
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Load the dataset
df = pd.read_csv(r"C:\Users\nithi\OneDrive\Desktop\ML PACKAGE FINAL F\readmission.csv")

# Shuffle the dataset only once before the split
df = shuffle(df, random_state=42)

# Split dataset into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Number of clients
num_clients = 5

# Calculate the split size for each client
split_size = len(X) // num_clients

# Define the client ID
client_id = int(input("Enter client number (1-5): ")) - 1

# Ensure unique indices for each client
start_index = client_id * split_size
end_index = start_index + split_size

# Sample data for the current client
X_client = X.iloc[start_index:end_index]
y_client = y.iloc[start_index:end_index]

# Show the distribution of labels for the current client
print(f"Client {client_id + 1} Labels Distribution:")
print(y_client.value_counts())

# Train the model using a linear kernel SVM
model = svm.SVC(kernel='linear')
model.fit(X_client, y_client)

# Calculate local accuracy
y_pred = model.predict(X_client)
local_accuracy = accuracy_score(y_client, y_pred)

print(f"Client {client_id + 1} local accuracy: {local_accuracy:.4f}")

# Prepare model weights to send to the server
model_weights = model.coef_

# Check if model_weights is None
if model_weights is None:
    print(f"Error: Model weights for Client {client_id + 1} are None.")
else:
    print(f"Client {client_id + 1} model weights: {model_weights}")

# Prepare the data to send
data_to_send = {
    'client_id': client_id + 1,
    'local_accuracy': f"{local_accuracy:.4f}",
    'client_size': len(X_client),  # Sending the size of data processed by the client
    'model_weights': model_weights  # Sending the model weights
}

# Socket setup
server_address = ('localhost', 5009)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(server_address)

# Sending the pickled data to the server
sock.sendall(pickle.dumps(data_to_send))
sock.close()
