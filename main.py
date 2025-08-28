from Process_data.data import data
from Clustering.algorithms.HDBSCAN import HDB
from Clustering.algorithms.kmeans import kMedoids
from Clustering.algorithms.spectral import Spectral
from Clustering.visualisation.SOM import SOM
from fms_and_ssq.fms import fms
from fms_and_ssq.ssq import ssq

def cluster(loaded_data, som = False):
    clustering_metrics_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/Clustering Metrics'
    data_labels = loaded_data.participantSessions[0].dataLabels

    print('Creating Models...\n')
    models = [HDB(loaded_data, data_labels), kMedoids(loaded_data, data_labels)]

    if som:
        print('Creating SOM...\n')
        som = SOM(loaded_data)

        for model in models:
            print(f'Plotting SOM for model {model.name}...\n')
            model.SOM_plot(som)
            print(f'Saving metrics for model {model.name}...\n')
            model.save_metrics(clustering_metrics_folder)
    else:
        for model in models:
            print(f'Saving metrics for model {model.name}...\n')
            model.save_metrics(clustering_metrics_folder)

def fms_ssq(loaded_data):
    fms_metrics_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/FMS Metrics'
    ssq_metrics_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/SSQ Metrics'
    fms_analysis = fms(loaded_data.fms, fms_metrics_folder)
    fms_analysis.analyse()
    ssq_analysis = ssq(loaded_data.ssq, ssq_metrics_folder)
    ssq_analysis.analyse()

def main():
    ITC_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/ITC Data'
    IEC_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/IEC Data'
    RBP_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/RBP Data'
    FMS_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/FMS Data'
    SSQ_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/SSQ Data'
    SOT_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/SOT Data'
    
    print('Processing Data...\n')
    loaded_data = data(ITC_folder, IEC_folder, RBP_folder, SSQ_folder, FMS_folder, SOT_folder)
    
    #cluster(loaded_data)

    fms_ssq(loaded_data)


if __name__ == "__main__":
    main()

    



