from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

class SOM:

    def __init__(self, data):
        self.data = data
        self.som = MiniSom(x = 10, y = 10, input_len = self.data.shape[1], sigma = 1.0, learning_rate = 0.5)
        self.som.random_weights_init(self.data)
        self.som.train_random(self.data, 1000)

        self.mapped = np.array([self.som.winner(x) for x in self.data])
        self.mappedX = [m[0] for m in self.mapped]
        self.mappedY = [m[1] for m in self.mapped]

    def plot(self, labels):
        plt.figure(figsize = (10,10))
        scatter = plt.scatter(self.mappedX, self.mappedY, c=labels, cmap='tab10', alpha = 0.7)
        plt.colorbar(scatter, label='Clusters')
        plt.title('Self-Organizing Map (SOM) Clustering')
        plt.savefig('Generated_plots/clustering/SOM_Clustering.png')
        plt.show()

