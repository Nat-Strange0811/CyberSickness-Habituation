from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

class SOM:

    def __init__(self, data):
        self.data = np.array(data.data)
        self.som = MiniSom(x = 50, y = 50, input_len = self.data.shape[1], sigma = 1.0, learning_rate = 0.5)
        self.som.random_weights_init(self.data)
        self.som.train_random(self.data, 1000)

        self.shapes = ['x', 'o', '+', "p", "<", "*"]

        self.mapped = np.array([self.som.winner(x) for x in self.data])
        self.mappedX = [m[0] for m in self.mapped]
        self.mappedY = [m[1] for m in self.mapped]

    def plot(self, labels, session_IDs, model_name):
        session_IDs = np.array(session_IDs)
        labels = np.array(labels)
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.figure(figsize = (10,10))
        for i, shape in enumerate(self.shapes):
            mask = (session_IDs == i)
            plt.scatter(
                np.array(self.mappedX)[mask],
                np.array(self.mappedY)[mask],
                marker=shape,
                c=np.array(labels)[mask],
                cmap='tab10',
                alpha = 0.7,
                label = f'Session {i+1}'
            )
        plt.colorbar(label='Cluster ID')
        plt.legend()
        plt.suptitle("SOM Clustering: colors = clusters, markers = sessions")
        plt.savefig(f'Generated_plots/clustering/SOM_Clustering_{model_name}_{date_time}.png')
        plt.close()

