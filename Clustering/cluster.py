from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from Clustering.visualisation.SOM import SOM
import pandas as pd
from scipy.stats import f_oneway, kruskal
import os
import csv

class cluster:
    def __init__ (self):
        self.silhouette = silhouette_score(self.data, self.labels)
        self.calinski = calinski_harabasz_score(self.data, self.labels)
        self.davies = davies_bouldin_score(self.data, self.labels)
        self.kruskal_results = {}
        self.kruskal()

    def kruskal(self):
        dataframe = pd.DataFrame(self.data, columns=self.dataLabels)
        dataframe['cluster'] = self.labels

        for feature in dataframe.columns[:-1]:
            groups = [dataframe[dataframe['cluster'] == c][feature] for c in dataframe['cluster'].unique()]
            h_stat, p_value = kruskal(*groups) if len(groups) > 1 else (None, None)
            self.kruskal_results[feature] = (h_stat, p_value)

    def SOM_plot(self, SOM):
        SOM.plot(self.labels)

    def save_metrics(self, filepath):
        labels = ["Model", "Silhouette", "Calinski", "Davies"] + self.data_labels
        metrics = [self.name, self.silhouette, self.calinski, self.davies] + [self.kruskal_results[label] for label in self.data_labels]

        csv = [labels, metrics]

        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        with open(os.path.join(filepath, f"{self.name}_metrics.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(csv)