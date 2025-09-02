import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import csv
import datetime

class fms():
    '''
    Class to perform FMS analysis
    '''
    def __init__(self, data, folder):
        '''
        Function Init

        Used to initialise the FMS analysis class.

        Inputs:

            Data - FMS data
            Folder - Folder to save results
        
        Outputs:

            Plot of FMS scores across sessions
        '''

        #Initialise class level variables
        self.save_folder = folder
        self.data = data
        self.markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "X", "P", "+", "x", "H", "d", "p"]
        self.colors = [
                    "#1f77b4", 
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                    "#bcbd22",
                    "#17becf",
                    "#e41a1c",
                    "#377eb8",
                    "#4daf4a",
                    "#984ea3",
                    "#00d0ff",
                    "#ffff00f3",
                ]

        #Initialise self.sessions dictionary to hold the list of all FMS scores from a particular session
        self.sessions = {
            self.sessionID: [] for self.sessionID in range(1, 7)
        }

        #Fill the self.sessions dictionary with FMS scores
        for participant in self.data.participantSessions:
            if not np.isnan(np.nanmean(participant.FMS['FMS'])):
                self.sessions[participant.sessionID].append(np.nanmean(participant.FMS['FMS']))

    def analyse(self):
        
        #Check for divergence between sessions
        self.anova()

        #Plot the distributions
        self.plot()

        #Perform t-test between session 1 and session 6
        self.t_test()

        #Save results
        self.save_results()

    def anova(self):
        '''
        Function to perform ANOVA test
        '''
        from scipy.stats import f_oneway
        #Make sure no nan values are present
        clean_sessions = [np.array(vals)[~np.isnan(vals)] for vals in self.sessions.values()]
        #Calculate f_stat and p_value
        f_stat, p_value = f_oneway(*clean_sessions)
        #Append result
        self.anova_results = (f_stat, p_value)
    
    def plot(self):
        '''
        Function to plot FMS scores across sessions
        '''

        #Initialise date time and figure size
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.figure(figsize=(20, 12))

        #First plot, each individual participants FMS scores over time
        plt.subplot(1,2,1)
        #Loop through each participant
        for i, part in enumerate(set([key[0] for key in self.data.fms.keys()])):
            #Hold individual session
            sessions = [[] for _ in range(6)]
            #Loop over each participant
            for participant in self.data.participantSessions:
                #Check if the participant is the correct one
                if participant.participant == i + 1:
                    #Check if the session FMS scores are not Nan
                    if not np.isnan(np.nanmean(participant.FMS['FMS'])):
                        #Append session FMS score to the list
                        sessions[participant.sessionID - 1].append(np.nanmean(participant.FMS['FMS']))
            #Calculate the mean value for each session
            sessions = [np.nanmean(session) for session in sessions]
            #Plot individual participant's FMS scores
            print(f"Participant {part} FMS Scores: {sessions}")
            plt.plot(range(1, 7), sessions, marker=self.markers[i], label=part, color=self.colors[i])
        plt.xlabel('Session')
        plt.ylabel('FMS Score')
        plt.legend()

        #Calculate the average for each session
        average = [np.nanmean(self.sessions[session]) for session in range(1, 7)]
        plt.subplot(1,2,2)
        plt.plot(range(1, 7), average, marker='o', color='black', label='Average')
        plt.xlabel('Session')
        plt.ylabel('FMS Score')
        plt.legend()
        plt.suptitle('FMS Scores Across Sessions')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'Generated_plots/fms/{date_time}_FMS_Scores_Across_Sessions.png')

    def t_test(self):
        #Perform a t_test compairing the first and last session
        clean_sessions = [np.array(vals)[~np.isnan(vals)] for vals in self.sessions.values()]
        t_stat, p_value = stats.ttest_ind(clean_sessions[0], clean_sessions[5], equal_var=False)
        self.t_test_results = (t_stat, p_value)

    def save_results(self):
        #Save results to a CSV
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        labels = ["Anova", "t_test"]
        metrics = [self.anova_results[1], self.t_test_results[1]]

        data = [labels, metrics]

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        with open(os.path.join(self.save_folder, f"fms_metrics_{date_time}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)