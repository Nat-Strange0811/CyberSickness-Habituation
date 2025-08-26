from Process_data.data import data
from Clustering.algorithms.HDBSCAN import HDB
from Clustering.algorithms.kmeans import kMedoids
from Clustering.algorithms.spectral import Spectral
from Clustering.visualisation.SOM import SOM


def main():
    ITC_folder = 'C:\Users\natty\OneDrive\Documents\Uni - Masters\Dissertation\Data\ITC Data'
    IEC_folder = 'C:\Users\natty\OneDrive\Documents\Uni - Masters\Dissertation\Data\IEC Data'
    RBP_folder = 'C:\Users\natty\OneDrive\Documents\Uni - Masters\Dissertation\Data\RBP Data'
    clustering_metrics_folder = 'C:\Users\natty\OneDrive\Documents\Uni - Masters\Dissertation\Data\Clustering Metrics'
    loaded_data = data(ITC_folder, IEC_folder, RBP_folder)
    data_labels = loaded_data[0].data_labels

    som = SOM(loaded_data.data)
    models = [HDB(loaded_data.data, data_labels), kMedoids(loaded_data.data, data_labels), Spectral(loaded_data.data, data_labels)]

    for model in models:
        model.SOM_plot(som)
        model.save_metrics(clustering_metrics_folder)


if __name__ == "__main__":
    main()

    



