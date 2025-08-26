from sklearn.cluster import SpectralClustering
from Clustering.cluster import cluster

class Spectral(cluster):

    def __init__(self, data, data_labels):
        self.data = data
        self.name = "Spectral Clustering"
        self.model = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
        self.labels = self.model.fit_predict(self.data)
        self.centroids = None
        self.data_labels = data_labels
        self.super().__init__()