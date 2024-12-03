#imporation des biblotheques nécessaire
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

#chargement du données
data = pd.read_csv("past_7_days.csv")
X = data[['latitude', 'longitude', 'mag']].values

# Function to calculate sum of squared distances (inertia) for a given k
#calcler la somme des ditances carrées (inertia) for
def calculate_inertia(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    return kmeans.inertia_

#fonction pour trouver le k optimal
def find_optimal_k(X, max_k=10):
    distortions = [calculate_inertia(X, k) for k in range(1, max_k + 1)]
    
    #afficher le curve du coude point
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()

#spécifier le nombre maximale 
max_clusters = 10

#trouver le k optimal on utilisant la methode du coude point
find_optimal_k(X, max_k=max_clusters)