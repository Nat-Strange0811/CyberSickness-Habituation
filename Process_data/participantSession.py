import os
import scipy.io
import numpy as np
import pandas as pd

class participantSession:
    def __init__(self, iec, itc, rbp, ssq_start, ssq_end, fms, sessionID, participant, runID):
        if runID:
            self.iec = iec
            self.itc = itc
            self.rbp = rbp
            self.SSQ_start = ssq_start
            self.SSQ_end = ssq_end
            self.FMS = fms
            self.sessionID = int(sessionID)
            self.participant = participant
            self.runID = runID
            self.dataLabels = [
                                'ITC-Fz-delta', 'ITC-Fz-theta1', 'ITC-Fz-theta2', 'ITC-Fz-alpha1', 'ITC-Fz-alpha2', 'ITC-Fz-alpha3', 'ITC-Fz-beta1', 'ITC-Fz-beta2', 'ITC-Fz-beta3', 'ITC-Fz-beta4',
                                'ITC-Cz-delta', 'ITC-Cz-theta1', 'ITC-Cz-theta2', 'ITC-Cz-alpha1', 'ITC-Cz-alpha2', 'ITC-Cz-alpha3', 'ITC-Cz-beta1', 'ITC-Cz-beta2', 'ITC-Cz-beta3', 'ITC-Cz-beta4',
                                'ITC-Pz-delta', 'ITC-Pz-theta1', 'ITC-Pz-theta2', 'ITC-Pz-alpha1', 'ITC-Pz-alpha2', 'ITC-Pz-alpha3', 'ITC-Pz-beta1', 'ITC-Pz-beta2', 'ITC-Pz-beta3', 'ITC-Pz-beta4',
                                'ITC-P3-delta', 'ITC-P3-theta1', 'ITC-P3-theta2', 'ITC-P3-alpha1', 'ITC-P3-alpha2', 'ITC-P3-alpha3', 'ITC-P3-beta1', 'ITC-P3-beta2', 'ITC-P3-beta3', 'ITC-P3-beta4',
                                'ITC-Cp5-delta', 'ITC-Cp5-theta1', 'ITC-Cp5-theta2', 'ITC-Cp5-alpha1', 'ITC-Cp5-alpha2', 'ITC-Cp5-alpha3', 'ITC-Cp5-beta1', 'ITC-Cp5-beta2', 'ITC-Cp5-beta3', 'ITC-Cp5-beta4',
                                'ITC-P4-delta', 'ITC-P4-theta1', 'ITC-P4-theta2', 'ITC-P4-alpha1', 'ITC-P4-alpha2', 'ITC-P4-alpha3', 'ITC-P4-beta1', 'ITC-P4-beta2', 'ITC-P4-beta3', 'ITC-P4-beta4',
                                'ITC-Cp6-delta', 'ITC-Cp6-theta1', 'ITC-Cp6-theta2', 'ITC-Cp6-alpha1', 'ITC-Cp6-alpha2', 'ITC-Cp6-alpha3', 'ITC-Cp6-beta1', 'ITC-Cp6-beta2', 'ITC-Cp6-beta3', 'ITC-Cp6-beta4',
                                'IEC-Fz_Pz-delta', 'IEC-Fz_Pz-theta1', 'IEC-Fz_Pz-theta2', 'IEC-Fz_Pz-alpha1', 'IEC-Fz_Pz-alpha2', 'IEC-Fz_Pz-alpha3', 'IEC-Fz_Pz-beta1', 'IEC-Fz_Pz-beta2', 'IEC-Fz_Pz-beta3', 'IEC-Fz_Pz-beta4',
                                'IEC-Fz_P3-delta', 'IEC-Fz_P3-theta1', 'IEC-Fz_P3-theta2', 'IEC-Fz_P3-alpha1', 'IEC-Fz_P3-alpha2', 'IEC-Fz_P3-alpha3', 'IEC-Fz_P3-beta1', 'IEC-Fz_P3-beta2', 'IEC-Fz_P3-beta3', 'IEC-Fz_P3-beta4',
                                'IEC-Fz_P4-delta', 'IEC-Fz_P4-theta1', 'IEC-Fz_P4-theta2', 'IEC-Fz_P4-alpha1', 'IEC-Fz_P4-alpha2', 'IEC-Fz_P4-alpha3', 'IEC-Fz_P4-beta1', 'IEC-Fz_P4-beta2', 'IEC-Fz_P4-beta3', 'IEC-Fz_P4-beta4',
                                'IEC-Fz_Cp5-delta', 'IEC-Fz_Cp5-theta1', 'IEC-Fz_Cp5-theta2', 'IEC-Fz_Cp5-alpha1', 'IEC-Fz_Cp5-alpha2', 'IEC-Fz_Cp5-alpha3', 'IEC-Fz_Cp5-beta1', 'IEC-Fz_Cp5-beta2', 'IEC-Fz_Cp5-beta3', 'IEC-Fz_Cp5-beta4',
                                'IEC-Fz_Cp6-delta', 'IEC-Fz_Cp6-theta1', 'IEC-Fz_Cp6-theta2', 'IEC-Fz_Cp6-alpha1', 'IEC-Fz_Cp6-alpha2', 'IEC-Fz_Cp6-alpha3', 'IEC-Fz_Cp6-beta1', 'IEC-Fz_Cp6-beta2', 'IEC-Fz_Cp6-beta3', 'IEC-Fz_Cp6-beta4',
                                'RBP-Fz-delta', 'RBP-Fz-theta', 'RBP-Fz-alpha', 'RBP-Fz-beta',
                                'RBP-Cz-delta', 'RBP-Cz-theta', 'RBP-Cz-alpha', 'RBP-Cz-beta', 'RBP-Cz-beta-theta-ratio',
                                'RBP-Pz-delta', 'RBP-Pz-theta', 'RBP-Pz-alpha', 'RBP-Pz-beta',
                                'RBP-P3-delta', 'RBP-P3-theta', 'RBP-P3-alpha', 'RBP-P3-beta',
                                'RBP-Cp5-delta', 'RBP-Cp5-theta', 'RBP-Cp5-alpha', 'RBP-Cp5-beta',
                                'RBP-P4-delta', 'RBP-P4-theta', 'RBP-P4-alpha', 'RBP-P4-beta',
                                'RBP-Cp6-delta', 'RBP-Cp6-theta', 'RBP-Cp6-alpha', 'RBP-Cp6-beta',
                                ]
            self.data = np.concatenate([self.itc, self.iec, self.rbp])
        else:
            length = iec.shape[0]
            self.SSQ_start = ssq_start
            self.SSQ_end = ssq_end
            self.FMS = fms
            self.sessionID = int(sessionID)
            self.participant = participant
            self.dataLabels = [
                                'ITC1-Fz-delta', 'ITC1-Fz-theta1', 'ITC1-Fz-theta2', 'ITC1-Fz-alpha1', 'ITC1-Fz-alpha2', 'ITC1-Fz-alpha3', 'ITC1-Fz-beta1', 'ITC1-Fz-beta2', 'ITC1-Fz-beta3', 'ITC1-Fz-beta4',
                                'ITC1-Cz-delta', 'ITC1-Cz-theta1', 'ITC1-Cz-theta2', 'ITC1-Cz-alpha1', 'ITC1-Cz-alpha2', 'ITC1-Cz-alpha3', 'ITC1-Cz-beta1', 'ITC1-Cz-beta2', 'ITC1-Cz-beta3', 'ITC1-Cz-beta4',
                                'ITC1-Pz-delta', 'ITC1-Pz-theta1', 'ITC1-Pz-theta2', 'ITC1-Pz-alpha1', 'ITC1-Pz-alpha2', 'ITC1-Pz-alpha3', 'ITC1-Pz-beta1', 'ITC1-Pz-beta2', 'ITC1-Pz-beta3', 'ITC1-Pz-beta4',
                                'ITC1-P3-delta', 'ITC1-P3-theta1', 'ITC1-P3-theta2', 'ITC1-P3-alpha1', 'ITC1-P3-alpha2', 'ITC1-P3-alpha3', 'ITC1-P3-beta1', 'ITC1-P3-beta2', 'ITC1-P3-beta3', 'ITC1-P3-beta4',
                                'ITC1-Cp5-delta', 'ITC1-Cp5-theta1', 'ITC1-Cp5-theta2', 'ITC1-Cp5-alpha1', 'ITC1-Cp5-alpha2', 'ITC1-Cp5-alpha3', 'ITC1-Cp5-beta1', 'ITC1-Cp5-beta2', 'ITC1-Cp5-beta3', 'ITC1-Cp5-beta4',
                                'ITC1-P4-delta', 'ITC1-P4-theta1', 'ITC1-P4-theta2', 'ITC1-P4-alpha1', 'ITC1-P4-alpha2', 'ITC1-P4-alpha3', 'ITC1-P4-beta1', 'ITC1-P4-beta2', 'ITC1-P4-beta3', 'ITC1-P4-beta4',
                                'ITC1-Cp6-delta', 'ITC1-Cp6-theta1', 'ITC1-Cp6-theta2', 'ITC1-Cp6-alpha1', 'ITC1-Cp6-alpha2', 'ITC1-Cp6-alpha3', 'ITC1-Cp6-beta1', 'ITC1-Cp6-beta2', 'ITC1-Cp6-beta3', 'ITC1-Cp6-beta4',
                                'IEC1-Fz_Pz-delta', 'IEC1-Fz_Pz-theta1', 'IEC1-Fz_Pz-theta2', 'IEC1-Fz_Pz-alpha1', 'IEC1-Fz_Pz-alpha2', 'IEC1-Fz_Pz-alpha3', 'IEC1-Fz_Pz-beta1', 'IEC1-Fz_Pz-beta2', 'IEC1-Fz_Pz-beta3', 'IEC1-Fz_Pz-beta4',
                                'IEC1-Fz_P3-delta', 'IEC1-Fz_P3-theta1', 'IEC1-Fz_P3-theta2', 'IEC1-Fz_P3-alpha1', 'IEC1-Fz_P3-alpha2', 'IEC1-Fz_P3-alpha3', 'IEC1-Fz_P3-beta1', 'IEC1-Fz_P3-beta2', 'IEC1-Fz_P3-beta3', 'IEC1-Fz_P3-beta4',
                                'IEC1-Fz_P4-delta', 'IEC1-Fz_P4-theta1', 'IEC1-Fz_P4-theta2', 'IEC1-Fz_P4-alpha1', 'IEC1-Fz_P4-alpha2', 'IEC1-Fz_P4-alpha3', 'IEC1-Fz_P4-beta1', 'IEC1-Fz_P4-beta2', 'IEC1-Fz_P4-beta3', 'IEC1-Fz_P4-beta4',
                                'IEC1-Fz_Cp5-delta', 'IEC1-Fz_Cp5-theta1', 'IEC1-Fz_Cp5-theta2', 'IEC1-Fz_Cp5-alpha1', 'IEC1-Fz_Cp5-alpha2', 'IEC1-Fz_Cp5-alpha3', 'IEC1-Fz_Cp5-beta1', 'IEC1-Fz_Cp5-beta2', 'IEC1-Fz_Cp5-beta3', 'IEC1-Fz_Cp5-beta4',
                                'IEC1-Fz_Cp6-delta', 'IEC1-Fz_Cp6-theta1', 'IEC1-Fz_Cp6-theta2', 'IEC1-Fz_Cp6-alpha1', 'IEC1-Fz_Cp6-alpha2', 'IEC1-Fz_Cp6-alpha3', 'IEC1-Fz_Cp6-beta1', 'IEC1-Fz_Cp6-beta2', 'IEC1-Fz_Cp6-beta3', 'IEC1-Fz_Cp6-beta4',
                                'RBP1-Fz-delta', 'RBP1-Fz-theta', 'RBP1-Fz-alpha', 'RBP1-Fz-beta',
                                'RBP1-Cz-delta', 'RBP1-Cz-theta', 'RBP1-Cz-alpha', 'RBP1-Cz-beta', 'RBP1-Cz-beta-theta-ratio',
                                'RBP1-Pz-delta', 'RBP1-Pz-theta', 'RBP1-Pz-alpha', 'RBP1-Pz-beta',
                                'RBP1-P3-delta', 'RBP1-P3-theta', 'RBP1-P3-alpha', 'RBP1-P3-beta',
                                'RBP1-Cp5-delta', 'RBP1-Cp5-theta', 'RBP1-Cp5-alpha', 'RBP1-Cp5-beta',
                                'RBP1-P4-delta', 'RBP1-P4-theta', 'RBP1-P4-alpha', 'RBP1-P4-beta',
                                'RBP1-Cp6-delta', 'RBP1-Cp6-theta', 'RBP1-Cp6-alpha', 'RBP1-Cp6-beta',
                                'ITC2-Fz-delta', 'ITC2-Fz-theta1', 'ITC2-Fz-theta2', 'ITC2-Fz-alpha1', 'ITC2-Fz-alpha2', 'ITC2-Fz-alpha3', 'ITC2-Fz-beta1', 'ITC2-Fz-beta2', 'ITC2-Fz-beta3', 'ITC2-Fz-beta4',
                                'ITC2-Cz-delta', 'ITC2-Cz-theta1', 'ITC2-Cz-theta2', 'ITC2-Cz-alpha1', 'ITC2-Cz-alpha2', 'ITC2-Cz-alpha3', 'ITC2-Cz-beta1', 'ITC2-Cz-beta2', 'ITC2-Cz-beta3', 'ITC2-Cz-beta4',
                                'ITC2-Pz-delta', 'ITC2-Pz-theta1', 'ITC2-Pz-theta2', 'ITC2-Pz-alpha1', 'ITC2-Pz-alpha2', 'ITC2-Pz-alpha3', 'ITC2-Pz-beta1', 'ITC2-Pz-beta2', 'ITC2-Pz-beta3', 'ITC2-Pz-beta4',
                                'ITC2-P3-delta', 'ITC2-P3-theta1', 'ITC2-P3-theta2', 'ITC2-P3-alpha1', 'ITC2-P3-alpha2', 'ITC2-P3-alpha3', 'ITC2-P3-beta1', 'ITC2-P3-beta2', 'ITC2-P3-beta3', 'ITC2-P3-beta4',
                                'ITC2-Cp5-delta', 'ITC2-Cp5-theta1', 'ITC2-Cp5-theta2', 'ITC2-Cp5-alpha1', 'ITC2-Cp5-alpha2', 'ITC2-Cp5-alpha3', 'ITC2-Cp5-beta1', 'ITC2-Cp5-beta2', 'ITC2-Cp5-beta3', 'ITC2-Cp5-beta4',
                                'ITC2-P4-delta', 'ITC2-P4-theta1', 'ITC2-P4-theta2', 'ITC2-P4-alpha1', 'ITC2-P4-alpha2', 'ITC2-P4-alpha3', 'ITC2-P4-beta1', 'ITC2-P4-beta2', 'ITC2-P4-beta3', 'ITC2-P4-beta4',
                                'ITC2-Cp6-delta', 'ITC2-Cp6-theta1', 'ITC2-Cp6-theta2', 'ITC2-Cp6-alpha1', 'ITC2-Cp6-alpha2', 'ITC2-Cp6-alpha3', 'ITC2-Cp6-beta1', 'ITC2-Cp6-beta2', 'ITC2-Cp6-beta3', 'ITC2-Cp6-beta4',
                                'IEC2-Fz_Pz-delta', 'IEC2-Fz_Pz-theta1', 'IEC2-Fz_Pz-theta2', 'IEC2-Fz_Pz-alpha1', 'IEC2-Fz_Pz-alpha2', 'IEC2-Fz_Pz-alpha3', 'IEC2-Fz_Pz-beta1', 'IEC2-Fz_Pz-beta2', 'IEC2-Fz_Pz-beta3', 'IEC2-Fz_Pz-beta4',
                                'IEC2-Fz_P3-delta', 'IEC2-Fz_P3-theta1', 'IEC2-Fz_P3-theta2', 'IEC2-Fz_P3-alpha1', 'IEC2-Fz_P3-alpha2', 'IEC2-Fz_P3-alpha3', 'IEC2-Fz_P3-beta1', 'IEC2-Fz_P3-beta2', 'IEC2-Fz_P3-beta3', 'IEC2-Fz_P3-beta4',
                                'IEC2-Fz_P4-delta', 'IEC2-Fz_P4-theta1', 'IEC2-Fz_P4-theta2', 'IEC2-Fz_P4-alpha1', 'IEC2-Fz_P4-alpha2', 'IEC2-Fz_P4-alpha3', 'IEC2-Fz_P4-beta1', 'IEC2-Fz_P4-beta2', 'IEC2-Fz_P4-beta3', 'IEC2-Fz_P4-beta4',
                                'IEC2-Fz_Cp5-delta', 'IEC2-Fz_Cp5-theta1', 'IEC2-Fz_Cp5-theta2', 'IEC2-Fz_Cp5-alpha1', 'IEC2-Fz_Cp5-alpha2', 'IEC2-Fz_Cp5-alpha3', 'IEC2-Fz_Cp5-beta1', 'IEC2-Fz_Cp5-beta2', 'IEC2-Fz_Cp5-beta3', 'IEC2-Fz_Cp5-beta4',
                                'IEC2-Fz_Cp6-delta', 'IEC2-Fz_Cp6-theta1', 'IEC2-Fz_Cp6-theta2', 'IEC2-Fz_Cp6-alpha1', 'IEC2-Fz_Cp6-alpha2', 'IEC2-Fz_Cp6-alpha3', 'IEC2-Fz_Cp6-beta1', 'IEC2-Fz_Cp6-beta2', 'IEC2-Fz_Cp6-beta3', 'IEC2-Fz_Cp6-beta4',
                                'RBP2-Fz-delta', 'RBP2-Fz-theta', 'RBP2-Fz-alpha', 'RBP2-Fz-beta',
                                'RBP2-Cz-delta', 'RBP2-Cz-theta', 'RBP2-Cz-alpha', 'RBP2-Cz-beta', 'RBP2-Cz-beta-theta-ratio',
                                'RBP2-Pz-delta', 'RBP2-Pz-theta', 'RBP2-Pz-alpha', 'RBP2-Pz-beta',
                                'RBP2-P3-delta', 'RBP2-P3-theta', 'RBP2-P3-alpha', 'RBP2-P3-beta',
                                'RBP2-Cp5-delta', 'RBP2-Cp5-theta', 'RBP2-Cp5-alpha', 'RBP2-Cp5-beta',
                                'RBP2-P4-delta', 'RBP2-P4-theta', 'RBP2-P4-alpha', 'RBP2-P4-beta',
                                'RBP2-Cp6-delta', 'RBP2-Cp6-theta', 'RBP2-Cp6-alpha', 'RBP2-Cp6-beta',
                                'ITC3-Fz-delta', 'ITC3-Fz-theta1', 'ITC3-Fz-theta2', 'ITC3-Fz-alpha1', 'ITC3-Fz-alpha2', 'ITC3-Fz-alpha3', 'ITC3-Fz-beta1', 'ITC3-Fz-beta2', 'ITC3-Fz-beta3', 'ITC3-Fz-beta4',
                                'ITC3-Cz-delta', 'ITC3-Cz-theta1', 'ITC3-Cz-theta2', 'ITC3-Cz-alpha1', 'ITC3-Cz-alpha2', 'ITC3-Cz-alpha3', 'ITC3-Cz-beta1', 'ITC3-Cz-beta2', 'ITC3-Cz-beta3', 'ITC3-Cz-beta4',
                                'ITC3-Pz-delta', 'ITC3-Pz-theta1', 'ITC3-Pz-theta2', 'ITC3-Pz-alpha1', 'ITC3-Pz-alpha2', 'ITC3-Pz-alpha3', 'ITC3-Pz-beta1', 'ITC3-Pz-beta2', 'ITC3-Pz-beta3', 'ITC3-Pz-beta4',
                                'ITC3-P3-delta', 'ITC3-P3-theta1', 'ITC3-P3-theta2', 'ITC3-P3-alpha1', 'ITC3-P3-alpha2', 'ITC3-P3-alpha3', 'ITC3-P3-beta1', 'ITC3-P3-beta2', 'ITC3-P3-beta3', 'ITC3-P3-beta4',
                                'ITC3-Cp5-delta', 'ITC3-Cp5-theta1', 'ITC3-Cp5-theta2', 'ITC3-Cp5-alpha1', 'ITC3-Cp5-alpha2', 'ITC3-Cp5-alpha3', 'ITC3-Cp5-beta1', 'ITC3-Cp5-beta2', 'ITC3-Cp5-beta3', 'ITC3-Cp5-beta4',
                                'ITC3-P4-delta', 'ITC3-P4-theta1', 'ITC3-P4-theta2', 'ITC3-P4-alpha1', 'ITC3-P4-alpha2', 'ITC3-P4-alpha3', 'ITC3-P4-beta1', 'ITC3-P4-beta2', 'ITC3-P4-beta3', 'ITC3-P4-beta4',
                                'ITC3-Cp6-delta', 'ITC3-Cp6-theta1', 'ITC3-Cp6-theta2', 'ITC3-Cp6-alpha1', 'ITC3-Cp6-alpha2', 'ITC3-Cp6-alpha3', 'ITC3-Cp6-beta1', 'ITC3-Cp6-beta2', 'ITC3-Cp6-beta3', 'ITC3-Cp6-beta4',
                                'IEC3-Fz_Pz-delta', 'IEC3-Fz_Pz-theta1', 'IEC3-Fz_Pz-theta2', 'IEC3-Fz_Pz-alpha1', 'IEC3-Fz_Pz-alpha2', 'IEC3-Fz_Pz-alpha3', 'IEC3-Fz_Pz-beta1', 'IEC3-Fz_Pz-beta2', 'IEC3-Fz_Pz-beta3', 'IEC3-Fz_Pz-beta4',
                                'IEC3-Fz_P3-delta', 'IEC3-Fz_P3-theta1', 'IEC3-Fz_P3-theta2', 'IEC3-Fz_P3-alpha1', 'IEC3-Fz_P3-alpha2', 'IEC3-Fz_P3-alpha3', 'IEC3-Fz_P3-beta1', 'IEC3-Fz_P3-beta2', 'IEC3-Fz_P3-beta3', 'IEC3-Fz_P3-beta4',
                                'IEC3-Fz_P4-delta', 'IEC3-Fz_P4-theta1', 'IEC3-Fz_P4-theta2', 'IEC3-Fz_P4-alpha1', 'IEC3-Fz_P4-alpha2', 'IEC3-Fz_P4-alpha3', 'IEC3-Fz_P4-beta1', 'IEC3-Fz_P4-beta2', 'IEC3-Fz_P4-beta3', 'IEC3-Fz_P4-beta4',
                                'IEC3-Fz_Cp5-delta', 'IEC3-Fz_Cp5-theta1', 'IEC3-Fz_Cp5-theta2', 'IEC3-Fz_Cp5-alpha1', 'IEC3-Fz_Cp5-alpha2', 'IEC3-Fz_Cp5-alpha3', 'IEC3-Fz_Cp5-beta1', 'IEC3-Fz_Cp5-beta2', 'IEC3-Fz_Cp5-beta3', 'IEC3-Fz_Cp5-beta4',
                                'IEC3-Fz_Cp6-delta', 'IEC3-Fz_Cp6-theta1', 'IEC3-Fz_Cp6-theta2', 'IEC3-Fz_Cp6-alpha1', 'IEC3-Fz_Cp6-alpha2', 'IEC3-Fz_Cp6-alpha3', 'IEC3-Fz_Cp6-beta1', 'IEC3-Fz_Cp6-beta2', 'IEC3-Fz_Cp6-beta3', 'IEC3-Fz_Cp6-beta4',
                                'RBP3-Fz-delta', 'RBP3-Fz-theta', 'RBP3-Fz-alpha', 'RBP3-Fz-beta',
                                'RBP3-Cz-delta', 'RBP3-Cz-theta', 'RBP3-Cz-alpha', 'RBP3-Cz-beta', 'RBP3-Cz-beta-theta-ratio',
                                'RBP3-Pz-delta', 'RBP3-Pz-theta', 'RBP3-Pz-alpha', 'RBP3-Pz-beta',
                                'RBP3-P3-delta', 'RBP3-P3-theta', 'RBP3-P3-alpha', 'RBP3-P3-beta',
                                'RBP3-Cp5-delta', 'RBP3-Cp5-theta', 'RBP3-Cp5-alpha', 'RBP3-Cp5-beta',
                                'RBP3-P4-delta', 'RBP3-P4-theta', 'RBP3-P4-alpha', 'RBP3-P4-beta',
                                'RBP3-Cp6-delta', 'RBP3-Cp6-theta', 'RBP3-Cp6-alpha', 'RBP3-Cp6-beta',
                                'ITC4-Fz-delta', 'ITC4-Fz-theta1', 'ITC4-Fz-theta2', 'ITC4-Fz-alpha1', 'ITC4-Fz-alpha2', 'ITC4-Fz-alpha3', 'ITC4-Fz-beta1', 'ITC4-Fz-beta2', 'ITC4-Fz-beta3', 'ITC4-Fz-beta4',
                                'ITC4-Cz-delta', 'ITC4-Cz-theta1', 'ITC4-Cz-theta2', 'ITC4-Cz-alpha1', 'ITC4-Cz-alpha2', 'ITC4-Cz-alpha3', 'ITC4-Cz-beta1', 'ITC4-Cz-beta2', 'ITC4-Cz-beta3', 'ITC4-Cz-beta4',
                                'ITC4-Pz-delta', 'ITC4-Pz-theta1', 'ITC4-Pz-theta2', 'ITC4-Pz-alpha1', 'ITC4-Pz-alpha2', 'ITC4-Pz-alpha3', 'ITC4-Pz-beta1', 'ITC4-Pz-beta2', 'ITC4-Pz-beta3', 'ITC4-Pz-beta4',
                                'ITC4-P3-delta', 'ITC4-P3-theta1', 'ITC4-P3-theta2', 'ITC4-P3-alpha1', 'ITC4-P3-alpha2', 'ITC4-P3-alpha3', 'ITC4-P3-beta1', 'ITC4-P3-beta2', 'ITC4-P3-beta3', 'ITC4-P3-beta4',
                                'ITC4-Cp5-delta', 'ITC4-Cp5-theta1', 'ITC4-Cp5-theta2', 'ITC4-Cp5-alpha1', 'ITC4-Cp5-alpha2', 'ITC4-Cp5-alpha3', 'ITC4-Cp5-beta1', 'ITC4-Cp5-beta2', 'ITC4-Cp5-beta3', 'ITC4-Cp5-beta4',
                                'ITC4-P4-delta', 'ITC4-P4-theta1', 'ITC4-P4-theta2', 'ITC4-P4-alpha1', 'ITC4-P4-alpha2', 'ITC4-P4-alpha3', 'ITC4-P4-beta1', 'ITC4-P4-beta2', 'ITC4-P4-beta3', 'ITC4-P4-beta4',
                                'ITC4-Cp6-delta', 'ITC4-Cp6-theta1', 'ITC4-Cp6-theta2', 'ITC4-Cp6-alpha1', 'ITC4-Cp6-alpha2', 'ITC4-Cp6-alpha3', 'ITC4-Cp6-beta1', 'ITC4-Cp6-beta2', 'ITC4-Cp6-beta3', 'ITC4-Cp6-beta4',
                                'IEC4-Fz_Pz-delta', 'IEC4-Fz_Pz-theta1', 'IEC4-Fz_Pz-theta2', 'IEC4-Fz_Pz-alpha1', 'IEC4-Fz_Pz-alpha2', 'IEC4-Fz_Pz-alpha3', 'IEC4-Fz_Pz-beta1', 'IEC4-Fz_Pz-beta2', 'IEC4-Fz_Pz-beta3', 'IEC4-Fz_Pz-beta4',
                                'IEC4-Fz_P3-delta', 'IEC4-Fz_P3-theta1', 'IEC4-Fz_P3-theta2', 'IEC4-Fz_P3-alpha1', 'IEC4-Fz_P3-alpha2', 'IEC4-Fz_P3-alpha3', 'IEC4-Fz_P3-beta1', 'IEC4-Fz_P3-beta2', 'IEC4-Fz_P3-beta3', 'IEC4-Fz_P3-beta4',
                                'IEC4-Fz_P4-delta', 'IEC4-Fz_P4-theta1', 'IEC4-Fz_P4-theta2', 'IEC4-Fz_P4-alpha1', 'IEC4-Fz_P4-alpha2', 'IEC4-Fz_P4-alpha3', 'IEC4-Fz_P4-beta1', 'IEC4-Fz_P4-beta2', 'IEC4-Fz_P4-beta3', 'IEC4-Fz_P4-beta4',
                                'IEC4-Fz_Cp5-delta', 'IEC4-Fz_Cp5-theta1', 'IEC4-Fz_Cp5-theta2', 'IEC4-Fz_Cp5-alpha1', 'IEC4-Fz_Cp5-alpha2', 'IEC4-Fz_Cp5-alpha3', 'IEC4-Fz_Cp5-beta1', 'IEC4-Fz_Cp5-beta2', 'IEC4-Fz_Cp5-beta3', 'IEC4-Fz_Cp5-beta4',
                                'IEC4-Fz_Cp6-delta', 'IEC4-Fz_Cp6-theta1', 'IEC4-Fz_Cp6-theta2', 'IEC4-Fz_Cp6-alpha1', 'IEC4-Fz_Cp6-alpha2', 'IEC4-Fz_Cp6-alpha3', 'IEC4-Fz_Cp6-beta1', 'IEC4-Fz_Cp6-beta2', 'IEC4-Fz_Cp6-beta3', 'IEC4-Fz_Cp6-beta4',
                                'RBP4-Fz-delta', 'RBP4-Fz-theta', 'RBP4-Fz-alpha', 'RBP4-Fz-beta',
                                'RBP4-Cz-delta', 'RBP4-Cz-theta', 'RBP4-Cz-alpha', 'RBP4-Cz-beta', 'RBP4-Cz-beta-theta-ratio',
                                'RBP4-Pz-delta', 'RBP4-Pz-theta', 'RBP4-Pz-alpha', 'RBP4-Pz-beta',
                                'RBP4-P3-delta', 'RBP4-P3-theta', 'RBP4-P3-alpha', 'RBP4-P3-beta',
                                'RBP4-Cp5-delta', 'RBP4-Cp5-theta', 'RBP4-Cp5-alpha', 'RBP4-Cp5-beta',
                                'RBP4-P4-delta', 'RBP4-P4-theta', 'RBP4-P4-alpha', 'RBP4-P4-beta',
                                'RBP4-Cp6-delta', 'RBP4-Cp6-theta', 'RBP4-Cp6-alpha', 'RBP4-Cp6-beta',
                                'ITC5-Fz-delta', 'ITC5-Fz-theta1', 'ITC5-Fz-theta2', 'ITC5-Fz-alpha1', 'ITC5-Fz-alpha2', 'ITC5-Fz-alpha3', 'ITC5-Fz-beta1', 'ITC5-Fz-beta2', 'ITC5-Fz-beta3', 'ITC5-Fz-beta4',
                                'ITC5-Cz-delta', 'ITC5-Cz-theta1', 'ITC5-Cz-theta2', 'ITC5-Cz-alpha1', 'ITC5-Cz-alpha2', 'ITC5-Cz-alpha3', 'ITC5-Cz-beta1', 'ITC5-Cz-beta2', 'ITC5-Cz-beta3', 'ITC5-Cz-beta4',
                                'ITC5-Pz-delta', 'ITC5-Pz-theta1', 'ITC5-Pz-theta2', 'ITC5-Pz-alpha1', 'ITC5-Pz-alpha2', 'ITC5-Pz-alpha3', 'ITC5-Pz-beta1', 'ITC5-Pz-beta2', 'ITC5-Pz-beta3', 'ITC5-Pz-beta4',
                                'ITC5-P3-delta', 'ITC5-P3-theta1', 'ITC5-P3-theta2', 'ITC5-P3-alpha1', 'ITC5-P3-alpha2', 'ITC5-P3-alpha3', 'ITC5-P3-beta1', 'ITC5-P3-beta2', 'ITC5-P3-beta3', 'ITC5-P3-beta4',
                                'ITC5-Cp5-delta', 'ITC5-Cp5-theta1', 'ITC5-Cp5-theta2', 'ITC5-Cp5-alpha1', 'ITC5-Cp5-alpha2', 'ITC5-Cp5-alpha3', 'ITC5-Cp5-beta1', 'ITC5-Cp5-beta2', 'ITC5-Cp5-beta3', 'ITC5-Cp5-beta4',
                                'ITC5-P4-delta', 'ITC5-P4-theta1', 'ITC5-P4-theta2', 'ITC5-P4-alpha1', 'ITC5-P4-alpha2', 'ITC5-P4-alpha3', 'ITC5-P4-beta1', 'ITC5-P4-beta2', 'ITC5-P4-beta3', 'ITC5-P4-beta4',
                                'ITC5-Cp6-delta', 'ITC5-Cp6-theta1', 'ITC5-Cp6-theta2', 'ITC5-Cp6-alpha1', 'ITC5-Cp6-alpha2', 'ITC5-Cp6-alpha3', 'ITC5-Cp6-beta1', 'ITC5-Cp6-beta2', 'ITC5-Cp6-beta3', 'ITC5-Cp6-beta4',
                                'IEC5-Fz_Pz-delta', 'IEC5-Fz_Pz-theta1', 'IEC5-Fz_Pz-theta2', 'IEC5-Fz_Pz-alpha1', 'IEC5-Fz_Pz-alpha2', 'IEC5-Fz_Pz-alpha3', 'IEC5-Fz_Pz-beta1', 'IEC5-Fz_Pz-beta2', 'IEC5-Fz_Pz-beta3', 'IEC5-Fz_Pz-beta4',
                                'IEC5-Fz_P3-delta', 'IEC5-Fz_P3-theta1', 'IEC5-Fz_P3-theta2', 'IEC5-Fz_P3-alpha1', 'IEC5-Fz_P3-alpha2', 'IEC5-Fz_P3-alpha3', 'IEC5-Fz_P3-beta1', 'IEC5-Fz_P3-beta2', 'IEC5-Fz_P3-beta3', 'IEC5-Fz_P3-beta4',
                                'IEC5-Fz_P4-delta', 'IEC5-Fz_P4-theta1', 'IEC5-Fz_P4-theta2', 'IEC5-Fz_P4-alpha1', 'IEC5-Fz_P4-alpha2', 'IEC5-Fz_P4-alpha3', 'IEC5-Fz_P4-beta1', 'IEC5-Fz_P4-beta2', 'IEC5-Fz_P4-beta3', 'IEC5-Fz_P4-beta4',
                                'IEC5-Fz_Cp5-delta', 'IEC5-Fz_Cp5-theta1', 'IEC5-Fz_Cp5-theta2', 'IEC5-Fz_Cp5-alpha1', 'IEC5-Fz_Cp5-alpha2', 'IEC5-Fz_Cp5-alpha3', 'IEC5-Fz_Cp5-beta1', 'IEC5-Fz_Cp5-beta2', 'IEC5-Fz_Cp5-beta3', 'IEC5-Fz_Cp5-beta4',
                                'IEC5-Fz_Cp6-delta', 'IEC5-Fz_Cp6-theta1', 'IEC5-Fz_Cp6-theta2', 'IEC5-Fz_Cp6-alpha1', 'IEC5-Fz_Cp6-alpha2', 'IEC5-Fz_Cp6-alpha3', 'IEC5-Fz_Cp6-beta1', 'IEC5-Fz_Cp6-beta2', 'IEC5-Fz_Cp6-beta3', 'IEC5-Fz_Cp6-beta4',
                                'RBP5-Fz-delta', 'RBP5-Fz-theta', 'RBP5-Fz-alpha', 'RBP5-Fz-beta',
                                'RBP5-Cz-delta', 'RBP5-Cz-theta', 'RBP5-Cz-alpha', 'RBP5-Cz-beta', 'RBP5-Cz-beta-theta-ratio',
                                'RBP5-Pz-delta', 'RBP5-Pz-theta', 'RBP5-Pz-alpha', 'RBP5-Pz-beta',
                                'RBP5-P3-delta', 'RBP5-P3-theta', 'RBP5-P3-alpha', 'RBP5-P3-beta',
                                'RBP5-Cp5-delta', 'RBP5-Cp5-theta', 'RBP5-Cp5-alpha', 'RBP5-Cp5-beta',
                                'RBP5-P4-delta', 'RBP5-P4-theta', 'RBP5-P4-alpha', 'RBP5-P4-beta',
                                'RBP5-Cp6-delta', 'RBP5-Cp6-theta', 'RBP5-Cp6-alpha', 'RBP5-Cp6-beta',
                                ]
            if length < 5:
                self.itc = self.extrapolate_repeats(itc).flatten()
                self.iec = self.extrapolate_repeats(iec).flatten()
                self.rbp = self.extrapolate_repeats(rbp).flatten()
                self.data = np.concatenate([self.itc, self.iec, self.rbp])
            elif length > 5:
                self.itc = itc[:5, :].flatten()
                self.iec = iec[:5, :].flatten()
                self.rbp = rbp[:5, :].flatten()
                self.data = np.concatenate([self.itc, self.iec, self.rbp])
            else:
                self.itc = itc.flatten()
                self.iec = iec.flatten()
                self.rbp = rbp.flatten()
                self.data = np.concatenate([self.itc, self.iec, self.rbp])

    def extrapolate_repeats(self, arr):
        n_repeats, n_samples = arr.shape
        x = np.arange(n_repeats)
        padded_arr = np.copy(arr)

        if n_repeats < 5:
            hold_arr = np.zeros((5-n_repeats, n_samples))
            for col in range(n_samples):
                # Fit a line to the existing repeats
                slope, intercept = np.polyfit(x, arr[:, col], 1)
                # Predict missing repeats
                for new_idx in range(0, 5 - n_repeats):
                    pred_value = slope * (new_idx + n_repeats) + intercept
                    hold_arr[new_idx, col] = pred_value
            padded_arr = np.vstack([padded_arr, hold_arr])
        return padded_arr
