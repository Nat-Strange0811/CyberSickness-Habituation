import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import csv
import datetime
from statsmodels.stats.multitest import multipletests
from scipy.stats import shapiro
from scipy.stats import wilcoxon

class sot():
    '''
    Class Sot to perform the analysis of the Sensory Organisation Test
    '''
    def __init__(self, data):
        '''
        Function Init

        Designed to initialise and trigger the analysis

        Inputs:

            data: The data to be analysed

        Outputs:

            Plots of all the SOT environments boxplots
            CSV t-test results for difference from before to after
        '''
        self.data = data
        self.t_test_results = {}
        self.environments = [
            "C1", "C2", "C3", "C4", "C5", "C6", "SOM", "VIS", "VEST", "PREF"
        ]
        self.analyse()

    def analyse(self):
        # Perform SOT analysis
        self.plot()

        self.t_test()

        self.save_results("C:/Users/natty/OneDrive/Documents/Uni - Masters/Dissertation/Data/SOT Metrics")

    def plot(self):
        '''
        Function - Plot

        Generates boxplots for each SOT environment

        Inputs:

            None

        Outputs:

            Boxplots saved to 'Generated_plots/sot'
        '''
        #Form the directory
        if not os.path.exists('Generated_plots/sot'):
            os.makedirs('Generated_plots/sot')

        
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        pre_data = []
        post_data = []
        labels = []

        # Collect all data across environments
        for environment in self.environments:
            data = self.data[environment]
            pre_data.append(data[0])   # initial scores
            post_data.append(data[1])  # final scores
            labels.append(environment)

        plt.figure(figsize=(12, 6))

        # Plot Pre distributions
        plt.subplot(1, 2, 1)
        plt.boxplot(pre_data, labels=labels)
        plt.title("Initial (Pre) SOT Scores")
        plt.ylabel("SOT Score")

        # Plot Post distributions
        plt.subplot(1, 2, 2)
        plt.boxplot(post_data, labels=labels)
        plt.title("Final (Post) SOT Scores")
        plt.ylabel("SOT Score")

        plt.suptitle("Comparing Distributions of SOT Scores Across Environments")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(f'Generated_plots/sot/{date_time}_SOT_distribution_All_Environments.png')
        plt.close()

    def t_test(self):
        '''
        Function - T-Test

        Performs independent t-tests for each SOT environment

        Inputs:

            None

        Outputs:

            t-test results stored in self.t_test_results
        '''
        #Initialise the p_values
        p_values = []

        #Loop over all environments
        for i, environment in enumerate(self.environments):
            stat, p = shapiro(self.data[environment][0])
            if p < 0.05:
                print(f"Not normal - {environment}")
                t_stat, p_value = wilcoxon(self.data[environment][0], self.data[environment][1])
            else:
                t_stat, p_value = stats.ttest_ind(self.data[environment][0], self.data[environment][1], equal_var=False)
            #Append the results
            self.t_test_results[environment] = [t_stat, p_value]
            p_values.append(p_value)
        #Correct the p_values using multiple tests
        _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')

        #Update the p_values in the results
        for i, environment in enumerate(self.environments):
            self.t_test_results[environment][1] = p_values_corrected[i]

    def save_results(self, filepath):
        '''
        Function Save Results

        Saves the results of the SOT analysis to a CSV file

        Inputs:

            filepath: The path to the file where results will be saved

        Outputs:

            CSV file
        '''

        #Initialise the date_time
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        #Form the data
        data = [["Environment","t_test", "p_value"]]
        for i, result in enumerate(self.t_test_results.values()):
            data.append([self.environments[i], result[0], result[1]])

        #Form directory and save
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        with open(os.path.join(filepath, f"SOT_metrics_{date_time}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)