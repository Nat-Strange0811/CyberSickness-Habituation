import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import csv
import datetime
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


class trends():
    def __init__(self, data, save_folder, split = False):
        '''
        Class - Trends

        Purpose:

            Runs the trend analysis, plotting the EEG Biomarkers against the control and FMS scores
            for each participant and overall averages. Isolates those which show significance.

        Inputs:

            data: loaded_data object, from which the individual sessions can be extracted
            split: whether the data has been split by runs or not
            save_folder: folder where the results will be saved
        '''
        self.plots = []
        self.eeg_p_values = []
        self.control_p_values = []
        self.eeg_r_values = []
        self.control_r_values = []
        self.data_labels = data.participantSessions[0].dataLabels
        self.save_folder = save_folder
        self.results = []
        self.column_titles = ["Participant", "EEG Biomarker", "EEG r", "EEG p", "Control r", "Control p"]
        self.data = data
        self.participant_data = {}
        self.split = split
        self.participants = set([participant.participant for participant in data.participantSessions])
        
        eeg_length = self.data.participantSessions[0].data.shape[0]

        self.fms_template_shape = self.data.participantSessions[0].FMS.shape
        self.fms_template = np.full(self.fms_template_shape, np.nan)

        #self.test()

        #Check if we have split the data
        if not self.split:
            #Set default shape for all participant data
            for participant in self.participants:
                self.participant_data[str(participant) + "_eeg"] = np.full((6, eeg_length), np.nan)
                self.participant_data[str(participant) + "_fms"] = np.tile(self.fms_template, (6, 1))
                self.participant_data[str(participant) + "_control"] = np.full((6, eeg_length), np.nan)
            #Iterate over data and replace dummy 'nan' data with realt data
            for participant in self.data.participantSessions:
                self.participant_data[str(participant.participant) + "_eeg"][participant.sessionID - 1] = participant.data
                try:
                    self.participant_data[str(participant.participant) + "_fms"][participant.sessionID - 1] = participant.FMS
                except:
                    self.participant_data[str(participant.participant) + "_fms"][participant.sessionID - 1] = np.nanmean(participant.FMS, axis = 0)
                self.participant_data[str(participant.participant) + "_control"][participant.sessionID - 1] = participant.control
            #Compute average across each session, need to use nanmean here, otherwise dummy nans not replaced will propogate
            self.average_sessions = {
                "eeg" : np.full((6,eeg_length), np.nan),
                "fms" : np.full((6,), np.nan),
                "control" : np.full((6,eeg_length), np.nan)
            }
            for i in range(6):
                for j in range(149):
                    self.average_sessions["eeg"][i][j] = np.nanmean([self.participant_data[str(part) + "_eeg"][i][j] for part in self.participants])
                    self.average_sessions["control"][i][j] = np.nanmean([self.participant_data[str(part) + "_control"][i][j] for part in self.participants])
                self.average_sessions["fms"][i] = np.nanmean([self.participant_data[str(part) + "_fms"][i][0] for part in self.participants])
            if eeg_length == 149:
                self.plot_runs()
                self.label = "Last_runs"
            else:
                self.plot_sessions()
                self.label = "Session_average"
        else:
            self.label = "All_runs"
            #See above, but shape has changed to incorporate the fact the data has been split by run
            for participant in self.participants:
                self.participant_data[str(participant) + "_eeg"] = np.full((30,149), np.nan)
                self.participant_data[str(participant) + "_fms"] = np.tile(self.fms_template, (30, 1))
                self.participant_data[str(participant) + "_control"] = np.full((30,149), np.nan)
            for participant in self.data.participantSessions:
                self.participant_data[str(participant.participant) + "_eeg"][(participant.sessionID - 1) * 5 + (participant.runID - 1)] = participant.data
                self.participant_data[str(participant.participant) + "_fms"][(participant.sessionID - 1) * 5 + (participant.runID - 1)] = participant.FMS
                self.participant_data[str(participant.participant) + "_control"][(participant.sessionID - 1) * 5 + (participant.runID - 1)] = participant.control
            self.average_sessions = {
                "eeg" : np.full((30,149), np.nan),
                "fms" : np.full((30,), np.nan),
                "control" : np.full((30,149), np.nan)
            }
            for i in range(30):
                for j in range(149):
                    self.average_sessions["eeg"][i][j] = np.nanmean([self.participant_data[str(part) + "_eeg"][i][j] for part in self.participants])
                    self.average_sessions["control"][i][j] = np.nanmean([self.participant_data[str(part) + "_control"][i][j] for part in self.participants])
                self.average_sessions["fms"][i] = np.nanmean([self.participant_data[str(part) + "_fms"][i][0] for part in self.participants])
            self.plot_runs()
        self.adjust_p_values()
        self.save_results()

    def adjust_p_values(self):

        eeg_reject, eeg_pvals_corrected, _, _ = multipletests(np.array(self.eeg_p_values), alpha=0.05, method='fdr_bh')
        control_reject, control_pvals_corrected, _, _ = multipletests(np.array(self.control_p_values), alpha=0.05, method='fdr_bh')

        for i in range(len(eeg_pvals_corrected)):
            self.results[i][2] = eeg_pvals_corrected[i]
            self.results[i][4] = control_pvals_corrected[i]

            if eeg_pvals_corrected[i] <= 0.05 and control_pvals_corrected[i] > 0.05:
                mode = "0"
            elif eeg_pvals_corrected[i] <= 0.05 and ((self.eeg_r_values[i] > 0 and self.control_r_values[i] < 0) or (self.eeg_r_values[i] < 0 and self.control_r_values[i] > 0)):
                mode = "1"
            else:
                continue
            eeg = self.plots[i][0]
            fms = self.plots[i][1]
            control = self.plots[i][2]
            biomarker = self.plots[i][3]
            participant = self.plots[i][4]
            stats = ((self.eeg_r_values[i], eeg_pvals_corrected[i]), (self.control_r_values[i], control_pvals_corrected[i]))
            self.plot_significant_results(eeg, fms, control, biomarker, participant, stats, mode)

    def plot_runs(self):
        '''
        Function designed to plot every single run experienced by the participant. It does this for
        both individual participants and the overall average. Is either triggered using all runs,
        or only the final one therefore the one 'closest' to the full dose of habituation.

        Inputs:

            Self - access to the class level information

        Outputs:

            Checks the significance of the result and plots them if EEG - FMS is significant and
            Control - FMS isn't.
        
        '''



        for eeg_biomarker in range(149):
        #Loop over all biomarkers so that we can compute the average trend for all participants
            eeg = []
            control = []
            #Initialise empty lists for the control and the eeg averages
            for datum in self.average_sessions["eeg"]:
            #Loop and append
                eeg.append(datum[eeg_biomarker])
            for datum in self.average_sessions["control"]:
            #Loop and append
                control.append(datum[eeg_biomarker])

            fms = self.average_sessions["fms"]
            #Retrieve the fms average

            stats = self.statistical_analysis(eeg, fms, control, eeg_biomarker, "average")
            #Run statistical test

            self.plots.append((eeg, fms, control, eeg_biomarker, "average", stats))

    def plot_sessions(self):
        '''
        Function is designed to plot every single session.

        Inputs:

            Self - Access to class level information

        Outputs:

            Calculates the stats and plots them if they are significant
    
        '''
    
        for eeg_biomarker in range(149):
            eeg = []
            control = []
            for datum in self.average_sessions["eeg"]:
                eeg.append(np.nanmean(
                    [datum[eeg_biomarker],
                    datum[eeg_biomarker + 149],
                    datum[eeg_biomarker + 298],
                    datum[eeg_biomarker + 447],
                    datum[eeg_biomarker + 596]]
                ))
            for datum in self.average_sessions["control"]:
                control.append(np.nanmean(
                    [datum[eeg_biomarker],
                    datum[eeg_biomarker + 149],
                    datum[eeg_biomarker + 298],
                    datum[eeg_biomarker + 447],
                    datum[eeg_biomarker + 596]]
                ))
            fms = self.average_sessions["fms"]

            stats = self.statistical_analysis(eeg, fms, control, eeg_biomarker, "average")

            self.plots.append((eeg, fms, control, eeg_biomarker, "average"))

    def statistical_analysis(self, eeg, fms, control, eeg_biomarker, participant):
        '''
        Function to calculate logistical regression relationships

        Inputs:

            Self - Class level information
            EEG - List of EEG values
            FMS - List of FMS values
            Control - List of Control values
            EEG_Biomarker - The specific EEG biomarker being analyzed
            Participant - The participant the data has come from

        Outputs:

            (eeg_r, eeg_p), (control_r, control_p) - Nested doubles, containing the values
        '''

        eeg_r, eeg_p = stats.pearsonr(eeg, fms)
        #Calculate eeg r and p value
        control_r, control_p = stats.pearsonr(control, fms)
        #Calculate control r and p values

        self.eeg_p_values.append(eeg_p)
        self.control_p_values.append(control_p)
        self.eeg_r_values.append(eeg_r)
        self.control_r_values.append(control_r)
        self.results.append([participant, eeg_biomarker, eeg_r, eeg_p, control_r, control_p])
        #Append the results of these tests to the results list
        return ((eeg_r, eeg_p), (control_r, control_p))
        #Return the values as tuple

    def plot_significant_results(self, eeg, fms, control, eeg_biomarker, participant, stats, plot_id):
        '''
        Function designed to plot the graphs so as to visualise the relationships

        Inputs:

            Self - object level information
            EEG - list of eeg values
            fms - list of fms values
            control - list of control values
            eeg_biomarker - what biomarker the data has come from
            participant - the participant the data has come from
            stats - the statistical results calculated previously
        
        Outputs:

            Generated plots for the significant results
        '''

        #Initialise the date and time
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot EEG + Control on the left y-axis
        ax1.plot(range(len(eeg)), eeg, marker='o', color='blue', label='EEG Biomarker')
        ax1.plot(range(len(control)), control, marker='o', color='green', label='Control')
        ax1.set_ylabel("EEG / Control", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Create a second y-axis for FMS
        ax2 = ax1.twinx()
        ax2.plot(range(len(fms)), fms, marker='o', color='orange', label='FMS')
        ax2.set_ylabel("FMS", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

        # Title
        plt.title(
            f'Participant {participant} - EEG Biomarker {self.data_labels[eeg_biomarker]}, '
            f'EEG_r: {stats[0][0]:.2f}, EEG_p: {stats[0][1]:.4f}, '
            f'Control_r: {stats[1][0]:.2f}, Control_p: {stats[1][1]:.4f}'
        )

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.tight_layout()

        if plot_id == "0":
            #Initialise directory
            if not os.path.exists(f'Generated_plots/trends/{self.label}/Significant vs Not Significant'):
                os.makedirs(f'Generated_plots/trends/{self.label}/Significant vs Not Significant')
            plt.savefig(f'Generated_plots/trends/{self.label}/Significant vs Not Significant/{date_time}_Participant_{participant}_EEG_Biomarker_{eeg_biomarker}.png')
        else:
            if not os.path.exists(f'Generated_plots/trends/{self.label}/Opposite correlation'):
                os.makedirs(f'Generated_plots/trends/{self.label}/Opposite correlation')
            plt.savefig(f'Generated_plots/trends/{self.label}/Opposite correlation/{date_time}_Participant_{participant}_EEG_Biomarker_{eeg_biomarker}.png')

        plt.close()
    def save_results(self):
        '''
        Function to save results to a csv file.

        Inputs:

            self - object level information
        
        Outputs:

            saved csv file
        '''
        #Initialise date time
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        #Form the data
        data = [self.column_titles]
        for result in self.results:
            data.append(result)

        #Initialise directory
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        #Write csv
        with open(os.path.join(self.save_folder, f"correlation_metrics_{self.label}_{date_time}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)