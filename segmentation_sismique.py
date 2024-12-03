#importation des bibliothéques
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Calcule de distance entre les centroids et les points de données
def euclidean_distance(data_point, centroids):
    return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

#Chargement des données depuis le fichier csv
data = pd.read_csv("past_7_days.csv")
X = data[['latitude', 'longitude', 'mag']].values

#Algorithme du K-Means
def K_means(x, k, max_iter=200):
    #Initialisation des points aléatoires
    centroids = x[np.random.choice(len(x), k, replace=False)]

    for _ in range(max_iter):
        #Calcul des distances et attribution des points au cluster le plus proche
        distances = np.vstack([euclidean_distance(point, centroids) for point in x])
        labels = np.argmin(distances, axis=1)
        
        #Mise à jour des centroids
        new_centroids = np.array([x[labels == i].mean(axis=0) for i in range(k)])

        #Vérification de la convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

#Spécification du nombre de clusters
k = 2
#Application du K-means clustering
labels, centroids = K_means(X, k)

#Ajouter une image d'arriere-plan
fig, ax = plt.subplots()
background_img = plt.imread("lat-lon-2.jpg")
ax.imshow(background_img, extent=[-180, 180, -90, 90], aspect='auto', alpha=0.5)
# Tracer les données et modifier la taille des cercles en fonction de la magnitud
plt.scatter(X[:, 1], X[:, 0], c=labels, s=(X[:, 2])**4, alpha=0.5, cmap='viridis', edgecolors='k')

# Tracer les centroids
plt.scatter(centroids[:, 1], centroids[:, 0], c='red', marker='*', s=200, label='Centroids')

# Modifier la fenêtre du plot
plt.title("Segmentation des données séismiques mondiales des 7 derniers jours avec K-Means")
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()

# Afficher la fenêtre du plot
plt.show()