from Clustering.cluster import cluster
import hdbscan 

class HDB(cluster):
    
    def __init__(self, data, data_labels):
        self.data = data.data
        self.participants = data.participantSessions
        self.sessions = [participant.sessionID for participant in self.participants]
        self.name = "HDBSCAN"
        self.model = hdbscan.HDBSCAN(min_cluster_size=6)
        self.labels = self.model.fit_predict(self.data)
        self.centroids = None
        self.data_labels = data_labels
        super().__init__()


