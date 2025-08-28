import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import csv

class sot():
    def __init__(self, data):
        self.data = data
        self.markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "X", "P", "+", "x", "H"]
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
                ]
        
        self.sessions = {}
        for session in range(1,7):
            self.sessions[session] = [np.mean(self.data[(participant, session)]['FMS']) 
                                             for participant in self.data.keys() 
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
        date_time = pd.to_datetime("now")
        plt.figure(figsize=(10, 6))
        for participant, i in enumerate(set([key[0] for key in self.data.keys()])):
            sessions = [self.data[(participant, session)] for session in range(1, 7) if (participant, session) in self.data]
            if len(sessions) < 5:
                continue
            plt.plot(range(1, len(sessions) + 1), sessions, marker=self.markers[i], label=participant, color=self.colors[i])
        average = np.mean([self.session_1, self.session_2, self.session_3, self.session_4, self.session_5, self.session_6], axis=1)
        plt.plot(range(1, 7), average, marker='o', color='black', label='Average')
        plt.title('FMS Scores Across Sessions')
        plt.xlabel('Session')
        plt.ylabel('FMS Score')
        plt.legend()
        plt.savefig(f'Generated_plots/fms/{date_time}_FMS_Scores_Across_Sessions.png')

    def t_test(self):
        t_stat, p_value = stats.ttest_ind(self.session_1, self.session_6, equal_var=False)
        self.t_test_results = (t_stat, p_value)

    def save_results(self, filepath):
        labels = ["Anova", "t_test"]
        metrics = [self.anova_results, self.t_test_results]

        data = [labels, metrics]

        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        with open(os.path.join(filepath, f"fms_metrics.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)