from Clustering.cluster import cluster
import hdbscan
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

class HDB(cluster):

    '''
    Class algorithm for HDBSCAN clustering

    This class implements the HDBSCAN clustering algorithm for a given loaded dataset. It identifies
    the optimal cluster size using the elbow method, returning the 'optimal' model.
    '''
    
    def __init__(self, data, data_labels):
        '''
        Init function for establishing object:

        Inputs:

        Data - Data object generated in main script, passed as entire object so all data can be accessed.
        Data_labels - List of labels corresponding to the data features.
        '''

        #Initialise data (EEG features), participants (the Participant Session objects) and sessions (what session Id each data point has)
        #sessions is not passed to the clustering algorithm so it is not biased.
        self.data = np.array(data.data)
        self.participants = data.participantSessions
        self.sessions = [participant.sessionID for participant in self.participants]
        self.FMS = [np.nanmean(participant.FMS['FMS']) for participant in self.participants]

        #Due to the structure of the data, some rows may contain NaN values. We need to remove these for the clustering algorithm to function.
        mask = ~np.isnan(self.data).any(axis=1)
        self.data = self.data[mask]
        self.sessions = np.array(self.sessions)[mask]
        self.FMS = np.array(self.FMS)[mask]

        #Initialise the name of the model, and call the elbow method to find the optimal cluster size
        self.name = "HDBSCAN"
        self.model = self.elbow()
        self.labels = self.model.fit_predict(self.data)

        #Filter out noise data points
        mask = self.labels != -1
        self.data = self.data[mask]
        self.sessions = self.sessions[mask]
        self.labels = self.labels[mask]
        self.FMS = self.FMS[mask]

        self.centroids = None
        self.data_labels = data_labels
        #Call the superclass constructor for graph generation and statistical analysis
        super().__init__()

    def elbow(self):
        '''
        Function - Elbow

        Uses the elbow method to identify the optimal cluster size, optimising for silhouette score.
        '''
        #Initialise needed variables
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        inertia = []
        best_score = -1
        best_k = None
        #Loop over a predefined range of cluster sizes
        for i in range(6, 15):
            #Create and fit the model
            hdb = hdbscan.HDBSCAN(min_cluster_size= i)
            hdb.fit(self.data)
            
            #Identify best score using silhouette score method
            score = silhouette_score(self.data, hdb.labels_)
            inertia.append(score)
            #Identify if this is the best score
            if score > best_score:
                best_score = score
                best_k = i

        print(best_k)
        # Plotting the elbow curve
        plt.figure(figsize=(8, 5))
        plt.plot(range(6, 15), inertia, marker='o')
        plt.title('Elbow Method for Optimal cluster size')
        plt.xlabel('Cluster size')
        plt.ylabel('Inertia')
        plt.grid()
        plt.savefig(f'Generated_plots/clustering/{date_time}_HDBSCAN_elbow_method.png')
        plt.close()

        # Return the model with optimal cluster size
        return hdbscan.HDBSCAN(min_cluster_size=best_k)
