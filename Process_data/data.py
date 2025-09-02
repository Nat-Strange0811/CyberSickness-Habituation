import os
from scipy.io import loadmat
from Process_data.participantSession import participantSession
import pandas as pd
import numpy as np

class data():
    '''
    Data class, stores all data from their relevant folders, held in one object to enable access to all relevant data
    '''
    def __init__(self, ITC_folder, IEC_folder, RBP_folder, SSQ_folder, FMS_folder, SOT_folder, split = False, stack = False):
        '''
        Function init

        Sets initial variables for the class

        Inputs:

            ITC_folder - Folder containing ITC data
            IEC_folder - Folder containing IEC data
            RBP_folder - Folder containing RBP data
            SSQ_folder - Folder containing SSQ data
            FMS_folder - Folder containing FMS data
            SOT_folder - Folder containing SOT data
            Split - Whether to split the data into seperate runs
            Stack - Whether to stack the runs or select only the final one

        Outputs:

            Data structure

        '''

        #Initialise the SOT dictionary, each environment has a 'pre' and 'post' list for statistical analysis
        self.SOT = {
            "C1": [[], []],
            "C2": [[], []],
            "C3": [[], []],
            "C4": [[], []],
            "C5": [[], []],
            "C6": [[], []],
            "SOM": [[], []],
            "VIS": [[], []],
            "VEST": [[], []],
            "PREF": [[], []]
        }

        #Initialise other variables
        self.split = split
        self.stack = stack
        self.errors = []

        #Initialise templates the eeg data corrupted for this session
        self.ITC_template = np.full((5, 70), np.nan)
        self.IEC_template = np.full((5, 50), np.nan)
        self.RBP_template = np.full((5, 29), np.nan)
        
        #Initialise FMS template
        self.FMS_template = pd.DataFrame(np.nan, 
                                         index=range(5), 
                                         columns= [
                                        "FMS", "Phosphenes", "Headache", "Neck pain", "Scalp pain",
                                        "Tingling", "Itching", "Burning sensation", "Sleepiness",
                                        "Trouble concentrating", "Acute mood change"
                                        ])

        #Initialise SSQ template
        self.SSQ_template = pd.DataFrame({
            "Variable" : [
                                "Stress",
                                "General Discomfort",
                                "Fatigue",
                                "Headache",
                                "Eye Strain",
                                "Difficulty Focusing",
                                "Increased Salivation",
                                "Sweating",
                                "Nausea",
                                "Difficulty Concentrating",
                                "Fullness of Head",
                                "Blurred Vision",
                                "Dizzy(Eyes Open)",
                                "Dizzy(Eyes Closed)",
                                "Vertigo",
                                "Stomach Awareness",
                                "Burping"
                            ],
            "Value" : [np.nan] * 17
        })

        #Initalise and form the data structures, relies on load_data to process individual participant sessions
        self.participantSessions = self.load_data(ITC_folder, IEC_folder, RBP_folder, SSQ_folder, FMS_folder, SOT_folder)
        self.ssq = {(session.participant, session.sessionID): (session.SSQ_start, session.SSQ_end) for session in self.participantSessions}
        self.fms = {(session.participant, session.sessionID): session.FMS for session in self.participantSessions}
        self.data = [session.data for session in self.participantSessions if session.data is not None]
        self.control = [session.control for session in self.participantSessions]

        print(self.errors)
    def load_data(self, ITC_folder, IEC_folder, RBP_folder, SSQ_folder, FMS_folder, SOT_folder):
        '''
        Function - Load Data 
        
        Iterates over all participants and sessions and crafts the appropriate participant session structure

        Inputs:

            ITC_folder: str - Path to the folder containing ITC data
            IEC_folder: str - Path to the folder containing IEC data
            RBP_folder: str - Path to the folder containing RBP data
            SSQ_folder: str - Path to the folder containing SSQ data
            FMS_folder: str - Path to the folder containing FMS data
            SOT_folder: str - Path to the folder containing SOT data

        Outputs:

            participantSessions: list - A list of participant session objects containing all relevant data

        '''

        #Initialise list to be returned
        participantSessions = []

        #Loop over all participants and sessions
        for participant in range(1,17):
            for session in range(1,7):
                #Initialise the paths to the relevant CSV's and Mats
                path_mat = f"P{participant}-0{session}.mat"
                path_csv = f"P{participant}_0{session}"

                #Load ITC data, if it doesn't exist hold the empty template
                try:
                    data = loadmat(os.path.join(ITC_folder, "Sliced",path_mat))
                    itc = data['output_itc'].transpose()
                except:
                    itc = self.ITC_template

                #Load IEC data, if it doesn't exist hold the empty template
                try:
                    data = loadmat(os.path.join(IEC_folder, "Sliced", path_mat))
                    iec = data['output_iec'].transpose()
                except:
                    iec = self.IEC_template

                #Load RBP data, if it doesn't exist hold the empty template
                try:
                    data = loadmat(os.path.join(RBP_folder, "Sliced", path_mat))
                    rbp = data['output_rbp'].transpose()
                except:
                    rbp = self.RBP_template

                #Load ITC control data, if it doesn't exist hold the empty template
                try:
                    data = loadmat(os.path.join(ITC_folder, "Control",path_mat))
                    control_itc = data['output_itc'].transpose()
                except:
                    control_itc = self.ITC_template

                #Load IEC control data, if it doesn't exist hold the empty template
                try:
                    data = loadmat(os.path.join(IEC_folder, "Control", path_mat))
                    control_iec = data['output_iec'].transpose()
                except:
                    control_iec = self.IEC_template

                #Load RBP control data, if it doesn't exist hold the empty template
                try:
                    data = loadmat(os.path.join(RBP_folder, "Control", path_mat))
                    control_rbp = data['output_rbp'].transpose()
                except:
                    control_rbp = self.RBP_template

                #Form the relevant path strings
                ssq_start = path_csv + '_SSQ_Start.csv'
                ssq_end = path_csv + '_SSQ_End.csv'
                fms = path_csv + '_FMS.csv'

                #Load ssq_start and end for the session, except for where one doesn't exist in which case load the empty sessions
                try:
                    ssq_start = pd.read_csv(os.path.join(SSQ_folder, ssq_start), header=None, names=["Variable", "Value"])
                    ssq_end = pd.read_csv(os.path.join(SSQ_folder, ssq_end), header=None, names=["Variable", "Value"])
                except:
                    ssq_start = self.SSQ_template
                    ssq_end = self.SSQ_template
                
                #Load fms data
                try:
                    fms = pd.read_csv(os.path.join(FMS_folder, fms))
                except:
                    fms = self.FMS_template

                #When splitting we need to seperate each run
                if self.split:
                    #Check the shapes of the control mats, and ammend if needed
                    control_iec, control_itc, control_rbp = self.check_shapes([control_iec, control_itc, control_rbp], [iec, itc, rbp])
                    for i, run in enumerate(itc):

                        #Make sure we don't have 'too many' runs, ie a double click has happened
                        if i > 4:
                            #Append an error message
                            self.errors.append(f"Skipping participant {participant}, session {session}, run {i + 1} due to excess runs.")
                            continue
                        #Craft the participant session
                        participant_session = participantSession(iec[i], run, rbp[i], control_iec[i], control_itc[i], control_rbp[i], ssq_start, ssq_end, fms.iloc[i], session, participant, i + 1, self.stack)
                        #If errors occured during the participant session, don't append it to the list
                        if participant_session.remove:
                            continue
                        participantSessions.append(participant_session)
                #otherwise, if not splitting the sessions, pass the full information to the participantSession class
                else:
                    participant_session = participantSession(iec, itc, rbp, control_iec, control_itc, control_rbp, ssq_start, ssq_end, fms, session, participant, None, self.stack)
                    if participant_session.remove:
                        continue
                    participantSessions.append(participant_session)
        
        #Read the SOT folder
        data = pd.read_csv(os.path.join(SOT_folder, "Dataset.csv"))
        #Iterate over the rows
        for idx, row in data.iterrows():
            #Identify the participant and their corresponding data
            participant = row['Participant']
            c1 = row['C1']
            c2 = row['C2']
            c3 = row['C3']
            c4 = row['C4']
            c5 = row['C5']
            c6 = row['C6']
            som = row['SOM']
            vis = row['VIS']
            vest = row['VEST']
            pref = row['PREF']
            #Check if the row corresponds to the start or end of a participant's session and append to the relevant list
            if "Start" in participant:
                self.SOT["C1"][0].append(c1)
                self.SOT["C2"][0].append(c2)
                self.SOT["C3"][0].append(c3)
                self.SOT["C4"][0].append(c4)
                self.SOT["C5"][0].append(c5)
                self.SOT["C6"][0].append(c6)
                self.SOT["SOM"][0].append(som)
                self.SOT["VIS"][0].append(vis)
                self.SOT["VEST"][0].append(vest)
                self.SOT["PREF"][0].append(pref)
            elif "End" in participant:
                self.SOT["C1"][1].append(c1)
                self.SOT["C2"][1].append(c2)
                self.SOT["C3"][1].append(c3)
                self.SOT["C4"][1].append(c4)
                self.SOT["C5"][1].append(c5)
                self.SOT["C6"][1].append(c6)
                self.SOT["SOM"][1].append(som)
                self.SOT["VIS"][1].append(vis)
                self.SOT["VEST"][1].append(vest)
                self.SOT["PREF"][1].append(pref)

        return participantSessions

    def check_shapes(self, controls, data):
        '''
        Function Check Shapes

        This function checks if the shapes of the control and data arrays match. If they don't, it pads the control array with NaNs to match the shape of the data array.

        Inputs:

            controls: A list of control arrays
            data: A list of data arrays
        
        Outputs:

            Padded control arrays
        '''
        #Loop over all data in control
        for i, control in enumerate(controls):
            #Check if the shapes match
            if control.shape != data[i].shape:
                #If they don't pad with the relevant amount of nans
                pad_length = data[i].shape[0] - control.shape[0]
                pad = np.full((pad_length, control.shape[1]), np.nan)
                control = np.vstack([control, pad])
                controls[i] = control
        #Return the padded control arrays
        return controls[0], controls[1], controls[2]