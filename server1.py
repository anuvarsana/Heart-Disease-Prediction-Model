import socket
import pickle
import numpy as np

# Function to aggregate the received model weights
def aggregate_models(client_updates, total_data_points):
    global_weights = None
    for client_data in client_updates:
        client_weights = client_data['model_weights']
        client_size = client_data['client_size']
        
        if global_weights is None:
            # Initialize global_weights with the same shape as client_weights
            global_weights = np.zeros_like(client_weights)
        
        # Weighted aggregation of the client model weights
        for key in range(len(client_weights)):
            global_weights[key] += client_weights[key] * client_size
    
    # Normalize by the total data points to get global model weights
    global_weights /= total_data_points
    return global_weights

# Start the server to receive updates from clients
def start_server():
    server_address = ('localhost', 5009)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(server_address)
    sock.listen(5)

    print(f"Server listenicng on {server_address[0]}:{server_address[1]}")

    client_updates = []
    total_data_points = 0

    while len(client_updates) < 5:
        connection, client_address = sock.accept()

        try:
            print(f"Connection from {client_address}")
            data = b""
            while True:
                packet = connection.recv(4096)
                if not packet:
                    break
                data += packet

            # Unpickle the received data from the client
            client_data = pickle.loads(data)
            
            # Print the received updates
            print(f"Received update from Client {client_data['client_id']}")
            print(f"Client {client_data['client_id']} local accuracy: {client_data['local_accuracy']}")
            print(f"Client {client_data['client_id']} sent model weights: {client_data['model_weights']}\n")

            # Store the client update
            client_updates.append(client_data)

            # Track total data points for weighted aggregation
            total_data_points += client_data['client_size']

        finally:
            connection.close()

    # Aggregate the models from all clients
    global_model_weights = aggregate_models(client_updates, total_data_points)
    
    print("Global model weights aggregated.")
    
    # Here you can compute the global accuracy by testing the global model on a test set.
    # For simplicity, let's just print a message for now:
    global_accuracy = np.mean([float(client_data['local_accuracy']) for client_data in client_updates])
    print(f"Aggregated global accuracy: {global_accuracy:.4f}")

# Start the server
if __name__ == "__main__":
    start_server()
