from sklearn.cluster import SpectralClustering
from Clustering.cluster import cluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler

class Spectral(cluster):
    '''
    Class used to initialise and trigger spectral clustering method
    '''
    def __init__(self, data, data_labels):
        '''
        Function - init:

        Used to initialise and provide access to the data and its labels.

        Inputs:

            Data - The data object created from main
            Data_labels - The labels corresponding to the data features

        Outputs:

            Spectral cluster
        '''

        #Initialise
        self.data = data.data
        self.data_scaled = StandardScaler().fit_transform(data.data)
        self.participants = data.participantSessions
        self.sessions = [participant.sessionID for participant in self.participants]
        self.name = "Spectral Clustering"
        self.model = self.elbow()
        self.labels = self.model.fit_predict(self.data_scaled)
        self.centroids = None
        self.data_labels = data_labels
        super().__init__()

    def elbow(self):
        best_score = -1
        best_k = None
        for i in range(2, 6):
            spectral = SpectralClustering(n_clusters=i, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
            spectral.fit(self.data_scaled)
            #Identify best score using silhouette score method
            score = silhouette_score(self.data, spectral.labels_)
            if score > best_score:
                best_score = score
                best_k = i
        
        # Return the model with optimal number of clusters
        return SpectralClustering(n_clusters=best_k, random_state=42)