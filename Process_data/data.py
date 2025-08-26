import os
from scipy.io import loadmat
from Process_data.participantSession import participantSession

class data():
    def __init__(self, ITC_folder, IEC_folder, RBP_folder):
        self.participantSessions = self.load_data(ITC_folder, IEC_folder, RBP_folder)
        self.data = [session.data for session in self.participantSessions]

    def load_data(self, ITC_folder, IEC_folder, RBP_folder, SSQ_folder, FMS_folder):
        participantSessions = []
        for session in os.listdir(ITC_folder):
            participant = session.split('_')[0]
            sessionID = session.split('_')[1].replace('.mat', '')

            itc = loadmat(os.path.join(ITC_folder, session))
            iec = loadmat(os.path.join(IEC_folder, session))
            rbp = loadmat(os.path.join(RBP_folder, session))

            for run, i in enumerate(itc):
                participantSession = participantSession(iec[i], itc[i], rbp[i], sessionID, participant, i + 1)
                participantSessions.append(participantSession)

        return participantSessions