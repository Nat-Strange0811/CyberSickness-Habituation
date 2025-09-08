import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import csv
import datetime
from scipy.stats import shapiro
import matplotlib.gridspec as gridspec


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

        self.remove_outliers = False

        #Fill the self.sessions dictionary with FMS scores
        for participant in self.data.participantSessions:
            if self.remove_outliers and (participant.participant == 5 or participant.participant == 11 or participant.participant == 14 or participant.participant == 15 or participant.participant == 16):
                continue
            if not np.isnan(np.nanmean(participant.FMS['FMS'])):
                self.sessions[participant.sessionID].append(np.nanmean(participant.FMS['FMS']))

        self.outliers = {
            self.sessionID: [] for self.sessionID in range(1, 7)
        }

        for participant in self.data.participantSessions:
            if (participant.participant == 5 or participant.participant == 11 or participant.participant == 14 or participant.participant == 15 or participant.participant == 16):
                continue
            if not np.isnan(np.nanmean(participant.FMS['FMS'])):
                self.outliers[participant.sessionID].append(np.nanmean(participant.FMS['FMS']))

        self.clean_sessions = [np.array(vals)[~np.isnan(vals)] for vals in self.sessions.values()]
    def analyse(self):

        #Perform t-test between session 1 and session 6
        self.t_test()
        
        #Plot the distributions
        self.plot()

        #Save results
        self.save_results()
    
    def plot(self):
        '''
        Function to plot FMS scores across sessions
        '''

        #Initialise date time and figure size
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.figure(figsize=(12, 10))  # adjust figure height

        # Define a 2-row grid, first row twice the height of second
        gs = gridspec.GridSpec(2, 4, height_ratios=[4, 3])
        '''
        #First plot, each individual participants FMS scores over time
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
            if i+1 == 5 or i+1 == 11 or i+1 == 14 or i+1 == 15 or i+1 == 16:
                plt.plot(range(1, 7), sessions, marker= '*', color = 'grey')
            else:
                plt.plot(range(1, 7), sessions, marker= '*', color = 'black')
        
        plt.plot([], [], color = 'grey', marker= '*', label = 'Outlier Participants')
        plt.plot([], [], color = 'black', marker= '*', label = 'Non Outlier Participants')
        '''

        #Calculate the average for each session
        average = [np.nanmean(self.sessions[session]) for session in range(1, 7)]
        errors = [np.nanstd(self.sessions[session]) / np.sqrt(len(self.sessions[session])) for session in range(1, 7)]
        #outliers = [np.nanmean(self.outliers[session]) for session in range(1, 7)]
        
        ax1 = plt.subplot(gs[0, :])
        ax1.errorbar(range(1, 7), average, yerr=errors, marker='o', color='black', label='Average')
        #plt.plot(range(1, 7), outliers, marker='o', color='magenta', label='Average (Outliers Removed)')
        ax1.set_xlabel("Session", fontsize=16, labelpad=10)
        ax1.set_ylabel("FMS Score (mean ± SEM)", fontsize=16, labelpad=10)
        ax1.legend(fontsize='large')
        ax1.grid(True, which="major", axis="y", linestyle="--", alpha=0.7)
        ax1.set_title('FMS Scores Across Sessions', fontsize = 18)

        #And plot the boxplot of the two
        ax2 = plt.subplot(gs[1, 1:3])
        sns.violinplot(data=[self.clean_sessions[0], self.clean_sessions[5]], palette="Set2", ax=ax2)
        sns.stripplot(data=[self.clean_sessions[0], self.clean_sessions[5]], 
              color="black", size=4, jitter=True, ax=ax2)
        ax2.set_xlabel("")  # leave blank since violin plot only needs xticks
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["Session 1", "Session 6"], fontsize = 16)
        ax2.set_ylabel("FMS Score", fontsize=16, labelpad=10)
        ax2.set_ylabel('FMS Score')
        ax2.set_title('FMS Score Distribution: Session 1 vs Session 6', fontsize = 18)

        title = "Graph Showing Average FMS Across Sessions, \nand Distribution of Session 1 vs 6"
        plt.suptitle(title, y=0.98, fontsize=22, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'Generated_plots/fms/{date_time}_FMS_Scores_Across_Sessions.png')
    
    def t_test(self):
        stat, p = shapiro(self.clean_sessions[0])
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #Perform a t_test compairing the first and last session
        if p < 0.05:
            from scipy.stats import wilcoxon
            t_stat, p_value = wilcoxon(self.clean_sessions[0], self.clean_sessions[5])
            print("Not normal")
        else:
            t_stat, p_value = stats.ttest_ind(self.clean_sessions[0], self.clean_sessions[5], equal_var=False)
        degrees_of_freedom = len(self.clean_sessions[0]) - 1
        self.t_test_results = (degrees_of_freedom, t_stat, p_value)

        print("Session 1 Mean:", np.nanmean(self.sessions[1]))
        print("Session 1 SD:", np.nanstd(self.sessions[1]))
        print("Session 6 Mean:", np.nanmean(self.sessions[6]))
        print("Session 6 SD:", np.nanstd(self.sessions[6]))

    def save_results(self):
        #Save results to a CSV
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        labels = ["Degrees of Freedom","t_value", "p_value"]

        data = [labels, self.t_test_results]

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        with open(os.path.join(self.save_folder, f"fms_metrics_{date_time}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)