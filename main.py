from Process_data.data import data
from Clustering.algorithms.HDBSCAN import HDB
from Clustering.algorithms.kmeans import kMedoids
from Clustering.algorithms.spectral import Spectral
from Clustering.visualisation.SOM import SOM
from fms_and_ssq.fms import fms
from fms_and_ssq.ssq import ssq
from Trends.trends import trends
from sot.sot import sot

#Function to launch SOT Analysis
def sot_analysis(sot_data):
    sot(sot_data)

#Function to launch the Cluster Analysis
def cluster(loaded_data):
    clustering_metrics_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/Clustering Metrics'
    data_labels = loaded_data.participantSessions[0].dataLabels

    models = [HDB(loaded_data, data_labels), kMedoids(loaded_data, data_labels)]

    for model in models:
        model.save_metrics(clustering_metrics_folder)

#Function to launch the FMS/SSQ analysis
def fms_ssq(loaded_data):
    fms_metrics_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/FMS Metrics'
    ssq_metrics_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/SSQ Metrics'
    fms_analysis = fms(loaded_data, fms_metrics_folder)
    fms_analysis.analyse()
    ssq_analysis = ssq(loaded_data.ssq, ssq_metrics_folder)
    ssq_analysis.analyse()

#Function to launch the trends analysis
def analyse_trends(loaded_data, split):
    trends_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/Trends'    
    trends(loaded_data, trends_folder, split=split)

#Main function which decideds which functions to use/not use, creates the data structure
def main():
    ITC_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/ITC Data'
    IEC_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/IEC Data'
    RBP_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/RBP Data'
    FMS_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/FMS Data'
    SSQ_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/SSQ Data'
    SOT_folder = 'C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/SOT Data'
    split = True


    loaded_data = data(ITC_folder, IEC_folder, RBP_folder, SSQ_folder, FMS_folder, SOT_folder, split = split, stack = False)

    cluster(loaded_data)

    #fms_ssq(loaded_data)

    #analyse_trends(loaded_data, split)

    #sot_analysis(loaded_data.SOT)


if __name__ == "__main__":
    main()

    



