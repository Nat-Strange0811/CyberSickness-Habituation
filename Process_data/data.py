import os
from scipy.io import loadmat
from Process_data.participantSession import participantSession
import pandas as pd

class data():
    def __init__(self, ITC_folder, IEC_folder, RBP_folder, SSQ_folder, FMS_folder, SOT_folder, split = False):
        self.SOT = {}
        self.split = split
        self.participantSessions = self.load_data(ITC_folder, IEC_folder, RBP_folder, SSQ_folder, FMS_folder, SOT_folder)
        self.ssq = {(session.participant, session.sessionID): (session.SSQ_start, session.SSQ_end) for session in self.participantSessions}
        self.fms = {(session.participant, session.sessionID): session.FMS for session in self.participantSessions}
        self.data = [session.data for session in self.participantSessions]

    def load_data(self, ITC_folder, IEC_folder, RBP_folder, SSQ_folder, FMS_folder, SOT_folder):
        participantSessions = []
        for session in os.listdir(ITC_folder):
            if session == 'Control':
                continue
            participant = session.split('-')[0]
            sessionID = session.split('-')[1].replace('.mat', '')

            data = loadmat(os.path.join(ITC_folder, session))
            itc = data['output_itc'].transpose()
            data = loadmat(os.path.join(IEC_folder, session))
            iec = data['output_iec'].transpose()
            data = loadmat(os.path.join(RBP_folder, session))
            rbp = data['output_rbp'].transpose()

            ssq_start = participant + '_' + sessionID + '_SSQ_Start.csv'
            ssq_end = participant + '_' + sessionID + '_SSQ_End.csv'
            fms = participant + '_' + sessionID + '_FMS.csv'

            ssq_start = pd.read_csv(os.path.join(SSQ_folder, ssq_start), header=None, names=["Variable", "Value"])
            ssq_end = pd.read_csv(os.path.join(SSQ_folder, ssq_end), header=None, names=["Variable", "Value"])
            fms = pd.read_csv(os.path.join(FMS_folder, fms))

            if self.split:
                for i, run in enumerate(itc):
                    participant_session = participantSession(iec[i], run, rbp[i], ssq_start, ssq_end, fms, sessionID, participant, i + 1)
                    participantSessions.append(participant_session)
            else:
                participant_session = participantSession(iec, itc, rbp, ssq_start, ssq_end, fms, sessionID, participant, None)
                participantSessions.append(participant_session)

        #for file in os.listdir(SOT_folder):
            #participant = file.split('_')[0]
            #occurence = file.split('_')[1].replace('.csv', '')
            #sot = pd.read_csv(os.path.join(SOT_folder, file))

            #self.SOT[(participant, occurence)] = sot

        return participantSessions
