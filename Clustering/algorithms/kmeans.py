from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from Clustering.cluster import cluster
import pandas as pd
import datetime
import numpy as np

class kMedoids(cluster):
    '''
    Class designed to execute the kMedoids clustering algorithm
    '''
    def __init__(self, data, data_labels):
        '''
        Function - Init

        Used to initialise and provide access to the data and its labels.

        Inputs:

            Data - The data object created from main
            Data_labels - The labels corresponding to the data features

        Outputs:

            K-Medoids cluster
        '''
        #Initialise the data as an array and provide access to the participants and their sessions
        self.data = np.array(data.data)
        self.participants = data.participantSessions
        self.sessions = [participant.sessionID for participant in self.participants]
        self.FMS = [np.nanmean(participant.FMS['FMS']) for participant in self.participants]

        #Remove any rows with NaN values
        mask = ~np.isnan(self.data).any(axis=1)
        self.data = self.data[mask]
        self.sessions = np.array(self.sessions)[mask]
        self.FMS = np.array(self.FMS)[mask]

        #Initialise the cluster and relative attributes
        self.name = "K-Medoids"
        self.model = self.elbow()
        self.labels = self.model.fit_predict(self.data)
        self.centroids = self.model.cluster_centers_
        self.data_labels = data_labels
        super().__init__()

    def elbow(self):
        '''
        Function - Elbow

        Used to determine the optimal number of clusters for K-Medoids.

        Inputs:

            None

        Outputs:

            K-Medoids model with optimal number of clusters
        '''
        #Initialise variables
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