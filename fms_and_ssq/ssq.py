import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import csv
import datetime

class ssq():
    def __init__(self, data, folder):
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
        self.sessions = {}
        for session in range(1,7):
            self.sessions[session] = [np.mean(self.data[(participant, session)][1]['Value'] - self.data[(participant, session)][0]['Value']) 
                                             for participant in [key[0] for key in list(self.data.keys())] 
                                             if (participant, session) in self.data]
        self.session_1 = self.sessions[1]
        self.session_2 = self.sessions[2]
        self.session_3 = self.sessions[3]
        self.session_4 = self.sessions[4]
        self.session_5 = self.sessions[5]
        self.session_6 = self.sessions[6]

    def analyse(self):
        # Perform FMS analysis
        self.anova()

        self.plot()

        self.t_test()

        self.save_results()

    def anova(self):
        from scipy.stats import f_oneway

        f_stat, p_value = f_oneway(self.session_1, self.session_2, self.session_3, self.session_4, self.session_5, self.session_6)
        self.anova_results = (f_stat, p_value)
    
    def plot(self):
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.figure(figsize=(10, 6))
        plt.subplot(1,2,1)
        for i, participant in enumerate(set([key[0] for key in self.data.keys()])):
            sessions = [np.mean(self.data[(participant, session)][1]['Value'] - self.data[(participant, session)][0]['Value']) for session in range(1, 7) if (participant, session) in self.data]
            print(sessions)
            plt.plot(range(1, len(sessions) + 1), sessions, marker=self.markers[i], label=participant, color=self.colors[i])
        average = [np.mean(self.session_1), np.mean(self.session_2), np.mean(self.session_3), np.mean(self.session_4), np.mean(self.session_5), np.mean(self.session_6)]
        plt.xlabel('Session')
        plt.ylabel('SSQ Score')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(range(1, 7), average, marker='o', color='black', label='Average')
        plt.title('SSQ Scores Across Sessions')
        plt.xlabel('Session')
        plt.ylabel('SSQ Score')
        plt.legend()
        plt.savefig(f'Generated_plots/ssq/{date_time}_SSQ_Scores_Across_Sessions.png')

    def t_test(self):
        t_stat, p_value = stats.ttest_ind(self.session_1, self.session_6, equal_var=False)
        self.t_test_results = (t_stat, p_value)

    def save_results(self):
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        labels = ["Anova", "t_test"]
        metrics = [self.anova_results[1], self.t_test_results[1]]

        data = [labels, metrics]

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        with open(os.path.join(self.save_folder, f"ssq_metrics_{date_time}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)