#importation des bibliothéques
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

#Calcule de distance entre les centroids et les points de données
def euclidean_distance(data_point, centroids):
    return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

#Chargement des données depuis le fichier csv
data = pd.read_csv("past_7_days.csv")
X = data[['latitude', 'longitude', 'mag']].values

#Spécification du nombre de clusters
k = 2

#Application d'algorithme K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

#Ajouter l'image d'arriere-plan
fig, ax = plt.subplots()
background_img = plt.imread("lat-lon-2.jpg")
ax.imshow(background_img, extent=[-180, 180, -90, 90], aspect='auto', alpha=0.5)

#Tracer les données et modifier la taille des cercles en fonction de la magnitud
scatter = plt.scatter(X[:, 1], X[:, 0], c=labels, s=(X[:, 2])**4, alpha=0.5, cmap='viridis', edgecolors='k')

#Tracer les centroids
plt.scatter(centroids[:, 1], centroids[:, 0], c='red', marker='*', s=200, label='Centroids')

#Modifier la fenêtre du plot
plt.title("Segmentation des données séismiques mondiales des 7 derniers jours avec K-Means")
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()

# fficher la fenêtre du plot
plt.show()