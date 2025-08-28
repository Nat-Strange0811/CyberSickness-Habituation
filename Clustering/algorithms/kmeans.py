from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from Clustering.cluster import cluster
import pandas as pd
import datetime

class kMedoids(cluster):
    def __init__(self, data, data_labels):
        self.data = data.data
        self.participants = data.participantSessions
        self.sessions = [participant.sessionID for participant in self.participants]
        self.name = "K-Medoids"
        self.model = self.elbow()
        self.labels = self.model.fit_predict(self.data)
        self.centroids = self.model.cluster_centers_
        self.data_labels = data_labels
        super().__init__()

    def elbow(self):
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        inertia = []
        best_score = -1
        best_k = None
        for i in range(2, 6):
            kmedoids = KMedoids(n_clusters=i, random_state=42)
            kmedoids.fit(self.data)
            inertia.append(kmedoids.inertia_)
            #Identify best score using silhouette score method
            score = silhouette_score(self.data, kmedoids.labels_)
            if score > best_score:
                best_score = score
                best_k = i

        # Plotting the elbow curve
        plt.figure(figsize=(8, 5))
        plt.plot(range(2, 6), inertia, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.grid()
        plt.savefig(f'Generated_plots/clustering/{date_time}_kmedoids_elbow_method.png')
        plt.close()

        # Return the model with optimal number of clusters
        return KMedoids(n_clusters=best_k, random_state=42)