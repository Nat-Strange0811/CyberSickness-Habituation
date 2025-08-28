import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import csv
import datetime


class trends():
    def __init__(self, eeg, fms, control):
        self.eeg = eeg
        self.fms = fms
        self.control = control

        self.plot_sessions()
        self.plot_runs()
        self.save_results()

    def plot_sessions():
        