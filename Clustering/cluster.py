from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from Clustering.visualisation.SOM import SOM
import pandas as pd
from scipy.stats import f_oneway, kruskal
import os
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
from statsmodels.stats.multitest import multipletests

class cluster:
    '''
    Class Cluster

    Super class for clustering algorithms, calculates statistical metrics and plots distributions of 
    significant features.
    '''
    def __init__ (self):
        '''
        Function Init

        Used to initialise the super class, and calculates the initial statistical metrics, 
        then triggers the kruskal statistical test. To identify causative features.

        Inputs:

            None

        Outputs:

            Statistical tests
        '''

        #Calculate statistical metrics
        self.silhouette = silhouette_score(self.data, self.labels)
        self.calinski = calinski_harabasz_score(self.data, self.labels)
        self.davies = davies_bouldin_score(self.data, self.labels)
        #Initialise kruskal dictionary, then trigger analysis
        self.kruskal_results = {}
        self.kruskal()

    def kruskal(self):
        '''
        Function - Kruskal

        Performs the Kruskal-Wallis H-test for each feature in the dataset.

        Inputs:
            
            None

        Outputs:

            H-statistic and p-value for each feature.
            Graph of the distributions for each feature that is significant
        '''

        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        #Initialise the dataframe, with column headers being the data labels
        dataframe = pd.DataFrame(self.data, columns=self.data_labels)
        #Insert additional columns for session divergence and cluster labels
        dataframe['Session Divergence'] = self.sessions
        dataframe['FMS'] = self.FMS
        dataframe['cluster'] = self.labels

        #Create a column for all data points
        dataframe["all"] = "All Clusters"
        p_values = []

        #Iterate over all EEG features
        for feature in dataframe.columns[:-2]:
            #Calculate the groups based on which cluster they fall in
            groups = [dataframe[dataframe['cluster'] == c][feature] for c in dataframe['cluster'].unique()]
            #Perform the Kruskal-Wallis H-test
            h_stat, p_value = kruskal(*groups) if len(groups) > 1 else (None, None)
            self.kruskal_results[feature] = (h_stat, p_value)
            p_values.append(p_value)

        #Adjust p-values for multiple comparisons
        _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')

        #Iterate over all features again
        for i, feature in enumerate(dataframe.columns[:-2]):
            #Update the Kruskal results with the corrected p-values
            self.kruskal_results[feature] = (self.kruskal_results[feature][0], p_values_corrected[i])
            #Check if the feature has significant variance
            if p_values_corrected[i] is None or p_values_corrected[i] > 0.05:
                if feature in ["Session Divergence", "FMS"]:
                    pass
                else:
                    continue

            #Plot overlapped distribution for the clusters, attempting to visualise separation
            
            plt.figure(figsize=(7, 6))
            '''
            plt.subplot(2, 1, 1)
            sns.stripplot(x="all", y=feature, hue="cluster", data=dataframe,
                        jitter=True, dodge=False, alpha=0.6)
            plt.xlabel("")
            plt.ylabel(feature)
            '''
            palette = sns.color_palette("Set2", len(dataframe['cluster'].unique()))


            #Plot the violin plot for each identified cluster
            #plt.subplot(2, 1, 2)
            sns.violinplot(x="cluster", y=feature, data=dataframe, palette=palette, alpha=0.7)
            sns.stripplot(x="cluster", y=feature, data=dataframe, 
              color="black", size=2, jitter=True, alpha=0.4)
            plt.xlabel("Cluster")
            plt.ylabel(feature)

            #Add a title to the plot
            plt.suptitle(f"{self.name}\n Kruskal-Wallis Test for {feature}\n p={p_values_corrected[i]:.4f}, h={self.kruskal_results[feature][0]:.4f}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            #Save the plot
            if not os.path.exists(f'Generated_plots/clustering/kruskal_{self.name}_{date_time}'):
                os.makedirs(f'Generated_plots/clustering/kruskal_{self.name}_{date_time}')

            plt.savefig(f'Generated_plots/clustering/kruskal_{self.name}_{date_time}/{feature}_kruskal.png')
            plt.close()


    def SOM_plot(self, SOM):
        '''
        Triggers plotting of SOM
        '''
        SOM.plot(self.labels, self.sessions, self.name)

    def save_metrics(self, filepath):
        '''
        Function Save Metrics:

        Saves the clustering metrics to a CSV file.

        Inputs:

            Filepath to save to

        Outputs:

            Saved CSV
        '''
        
        

        #Add Session Divergence label
        self.data_labels.append("Session Divergence")
        self.data_labels.append("FMS")

        #Initialise the date/time and hold the clusters and labels
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        clusters = np.array(self.labels)
        cluster_labels = []
        for i, label in enumerate(sorted(set(self.labels))):
            cluster_labels.append(f"Cluster {i + 1}")

        #Calculate the average session number for each cluster
        cluster_average = [np.mean(np.array(self.sessions)[clusters == label]) for label in sorted(set(self.labels))]

        #Create the csv structure of labels and metrics
        labels = ["Model", "Silhouette", "Calinski", "Davies"] + list(cluster_labels) + self.data_labels
        metrics = [self.name, self.silhouette, self.calinski, self.davies] + cluster_average + [self.kruskal_results[label][1] for label in self.data_labels]
        h_values = ['H_values', '', '', '', '', ''] + [self.kruskal_results[label][0] for label in self.data_labels]

        data = zip(labels, metrics, h_values)

        #Write the csv
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        with open(os.path.join(filepath, f"{self.name}_metrics_{date_time}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)