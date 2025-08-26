from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from Clustering.cluster import cluster

class kMedoids(cluster):
    def __init__(self, data, data_labels):
        self.super().__init__()
        self.name = "K-Medoids"
        self.data = data
        self.model = self.elbow()
        self.labels = self.model.fit_predict(self.data)
        self.centroids = self.model.cluster_centers_
        self.data_labels = data_labels
        self.super().__init__()

    def elbow(self):
        inertia = []
        best_score = -1
        best_k = None
        for i in range(1, 6):
            kmeans = KMedoids(n_clusters=i, random_state=42)
            kmeans.fit(self.data)
            inertia.append(kmeans.inertia_)
            #Identify best score using silhouette score method
            score = silhouette_score(self.data, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = i

        # Plotting the elbow curve
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 6), inertia, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.grid()
        plt.savefig('Generated_plots/clustering/kmeans_elbow_method.png')
        plt.show()
        
        # Return the model with optimal number of clusters
        return KMedoids(n_clusters=best_k, random_state=42)