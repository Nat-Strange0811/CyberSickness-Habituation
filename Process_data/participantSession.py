import os
import scipy.io

class participantSession:
    def __init__(self, iec, itc, rbp, sessionID, participant, runID):
        self.iec = iec
        self.itc = itc
        self.rbp = rbp
        self.sessionID = sessionID
        self.participant = participant
        self.runID = runID
        self.dataLabels = ['ITC-Fz-delta', 'ITC-Fz-theta1', 'ITC-Fz-theta2', 'ITC-Fz-alpha1', 'ITC-Fz-alpha2', 'ITC-Fz-alpha3', 'ITC-Fz-beta1', 'ITC-Fz-beta2', 'ITC-Fz-beta3', 'ITC-Fz-beta4',
                           'ITC-Cz-delta', 'ITC-Cz-theta1', 'ITC-Cz-theta2', 'ITC-Cz-alpha1', 'ITC-Cz-alpha2', 'ITC-Cz-alpha3', 'ITC-Cz-beta1', 'ITC-Cz-beta2', 'ITC-Cz-beta3', 'ITC-Cz-beta4',
                           'ITC-Pz-delta', 'ITC-Pz-theta1', 'ITC-Pz-theta2', 'ITC-Pz-alpha1', 'ITC-Pz-alpha2', 'ITC-Pz-alpha3', 'ITC-Pz-beta1', 'ITC-Pz-beta2', 'ITC-Pz-beta3', 'ITC-Pz-beta4',
                           'ITC-P3-delta', 'ITC-P3-theta1', 'ITC-P3-theta2', 'ITC-P3-alpha1', 'ITC-P3-alpha2', 'ITC-P3-alpha3', 'ITC-P3-beta1', 'ITC-P3-beta2', 'ITC-P3-beta3', 'ITC-P3-beta4',
                           'ITC-Cp5-delta', 'ITC-Cp5-theta1', 'ITC-Cp5-theta2', 'ITC-Cp5-alpha1', 'ITC-Cp5-alpha2', 'ITC-Cp5-alpha3', 'ITC-Cp5-beta1', 'ITC-Cp5-beta2', 'ITC-Cp5-beta3', 'ITC-Cp5-beta4',
                           'ITC-P4-delta', 'ITC-P4-theta1', 'ITC-P4-theta2', 'ITC-P4-alpha1', 'ITC-P4-alpha2', 'ITC-P4-alpha3', 'ITC-P4-beta1', 'ITC-P4-beta2', 'ITC-P4-beta3', 'ITC-P4-beta4',
                           'ITC-Cp6-delta', 'ITC-Cp6-theta1', 'ITC-Cp6-theta2', 'ITC-Cp6-alpha1', 'ITC-Cp6-alpha2', 'ITC-Cp6-alpha3', 'ITC-Cp6-beta1', 'ITC-Cp6-beta2', 'ITC-Cp6-beta3', 'ITC-Cp6-beta4',
                            'IEC',
                            'RBP-Fz-delta', 'RBP-Fz-theta', 'RBP-Fz-alpha', 'RBP-Fz-beta',
                            'RBP-Cz-delta', 'RBP-Cz-theta', 'RBP-Cz-alpha', 'RBP-Cz-beta', 'RBP-Cz-beta/theta',
                            'RBP-Pz-delta', 'RBP-Pz-theta', 'RBP-Pz-alpha', 'RBP-Pz-beta',
                            'RBP-P3-delta', 'RBP-P3-theta', 'RBP-P3-alpha', 'RBP-P3-beta',
                            'RBP-Cp5-delta', 'RBP-Cp5-theta', 'RBP-Cp5-alpha', 'RBP-Cp5-beta',
                            'RBP-P4-delta', 'RBP-P4-theta', 'RBP-P4-alpha', 'RBP-P4-beta',
                            'RBP-Cp6-delta', 'RBP-Cp6-theta', 'RBP-Cp6-alpha', 'RBP-Cp6-beta',
                            ]
        self.data = self.itc + self.iec + self.rbp

