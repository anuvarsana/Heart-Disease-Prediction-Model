# Heart-Disease-Prediction-Model

**Readmission Prediction System**

This repository contains a Readmission Prediction System designed to predict hospital readmission using machine learning models. It incorporates client-server architecture, federated learning techniques, and a Streamlit-based interactive dashboard for model comparison.

**Features**

**1. Federated Learning Simulation:**

- A client-server setup where clients train local models and share updates with the server for global aggregation.
- Ensures data privacy as raw data remains on the clients.

**2. Machine Learning Models:**

- Random Forest, Decision Tree, and Support Vector Machine (SVM) for prediction.
- Hyperparameter tuning and overfitting detection included.

**3. Interactive Dashboard:**

- Built with Streamlit for real-time visualization and comparison of model performance.

**File Descriptions**

**1. client1.py (Client Script)**
- Each client trains an SVM model on a partition of the dataset.
- Computes local accuracy and shares model weights with the server.
  
- **Key Features:**
  - **Data Partitioning:** Divides data among clients based on their IDs.
  - **Model Training:** Uses SVM with a linear kernel.
  - **Communication:** Sends updates to the server via sockets.
  
**2. server1.py (Server Script)**
- Aggregates model weights received from clients to create a global model.
- Calculates aggregated global accuracy.
  
- **Key Features:**
  - **Weighted Aggregation:** Combines updates based on client data size.
  - **Model Aggregation:** Normalizes aggregated weights for global use.

**3. streamlitcapf.py (Streamlit Dashboard)**
- Provides an interactive interface for:
    -Training and testing different models (Random Forest, Decision Tree, SVM).
    -Hyperparameter tuning.
    -Detecting overfitting.
    -Visualizing model accuracy comparisons.
-Includes preprocessing steps like feature scaling and encoding.

