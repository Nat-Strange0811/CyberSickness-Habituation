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

class cluster:
    def __init__ (self):
        self.silhouette = silhouette_score(self.data, self.labels)
        self.calinski = calinski_harabasz_score(self.data, self.labels)
        self.davies = davies_bouldin_score(self.data, self.labels)
        self.kruskal_results = {}
        self.kruskal()

    def kruskal(self):
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataframe = pd.DataFrame(self.data, columns=self.data_labels)
        dataframe['Session Divergence'] = self.sessions
        dataframe['cluster'] = self.labels

        dataframe["all"] = 1

        for feature in dataframe.columns[:-2]:
            groups = [dataframe[dataframe['cluster'] == c][feature] for c in dataframe['cluster'].unique()]
            h_stat, p_value = kruskal(*groups) if len(groups) > 1 else (None, None)
            self.kruskal_results[feature] = (h_stat, p_value)
            plt.figure(figsize=(14, 5))

            plt.subplot(1, 2, 1)
            sns.stripplot(x="all", y=feature, hue="cluster", data=dataframe,
                        jitter=True, dodge=True, alpha=0.6)
            plt.xlabel("")
            plt.ylabel(feature)

            plt.subplot(1, 2, 2)
            sns.violinplot(x="cluster", y=feature, data=dataframe)
            sns.stripplot(x="cluster", y=feature, data=dataframe, 
              color="black", size=2, jitter=True, alpha=0.4)
            plt.xlabel("Cluster")
            plt.ylabel(feature)

            plt.suptitle(f"Kruskal-Wallis Test for {feature}\n p={p_value:.4f}, h={h_stat:.4f}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if not os.path.exists(f'Generated_plots/clustering/kruskal_{self.name}_{date_time}'):
                os.makedirs(f'Generated_plots/clustering/kruskal_{self.name}_{date_time}')

            plt.savefig(f'Generated_plots/clustering/kruskal_{self.name}_{date_time}/{feature}_kruskal.png')
            plt.close()


    def SOM_plot(self, SOM):
        SOM.plot(self.labels, self.sessions, self.name)

    def save_metrics(self, filepath):
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(self.labels)
        clusters = np.array(self.labels)
        cluster_labels = sorted(set(self.labels))

        cluster_average = [np.mean(np.array(self.sessions)[clusters == label]) for label in cluster_labels]

        labels = ["Model", "Silhouette", "Calinski", "Davies"] + list(cluster_labels) + self.data_labels
        metrics = [self.name, self.silhouette, self.calinski, self.davies] + cluster_average + [self.kruskal_results[label][1] for label in self.data_labels]

        data = [labels, metrics]

        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        with open(os.path.join(filepath, f"{self.name}_metrics_{date_time}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)