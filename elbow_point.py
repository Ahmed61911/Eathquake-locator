import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load your data
data = pd.read_csv("past_7_days.csv")
X = data[['latitude', 'longitude', 'mag']].values

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# Function to assign each point to the closest centroid
def assign_to_clusters(X, centroids):
    distances = np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in X])
    return np.argmin(distances, axis=1)

# Function to update centroids based on the mean of assigned points
def update_centroids(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

# Function to calculate sum of squared distances (inertia)
def calculate_inertia(X, centroids, labels):
    return np.sum([euclidean_distance(X[i], centroids[labels[i]])**2 for i in range(len(X))])

# Function to perform K-Means clustering
def k_means(X, k, max_iter=200):
    # Initialize centroids
    centroids = X[np.random.choice(len(X), k, replace=False)]
    
    for _ in range(max_iter):
        # Assign points to clusters
        labels = assign_to_clusters(X, centroids)
        
        # Update centroids
        new_centroids = update_centroids(X, labels, k)
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    # Calculate and return inertia
    inertia = calculate_inertia(X, centroids, labels)
    return inertia

# Function to determine the optimal k using the elbow method
def find_optimal_k(X, max_k=10):
    inertias = [k_means(X, k) for k in range(1, max_k + 1)]
    
    # Plot the elbow curve
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()

# Specify the maximum number of clusters to consider
max_clusters = 10

# Find the optimal k using the elbow method
find_optimal_k(X, max_k=max_clusters)
