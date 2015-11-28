'''
k-Means Clustering
Author: Sam Boysel
Algorithm 13.5, page 335.
'''
import random
import numpy as np
from pprint import pprint


class KMeans():

    def __init__(self, k, data):
        assert isinstance(k, int)
        assert isinstance(data, np.ndarray)
        self.k = k
        self.data = data
        self.centers = {i: np.random.randn(data.shape[1])
                        for i in range(1, k+1)}
        self.members = {i: [] for i in range(1, k+1)}
        self.iterations = 0
        self.mse = 0.

    def print_centers(self):
        pprint(self.centers)

    def cluster(self, epsilon=0.001):
        delta = 1.
        while epsilon <= delta:
            # Cluster Assignment
            for x in self.data:
                # print(x)
                dists = {}
                for i, c in self.centers.items():
                    d = np.linalg.norm(x - c) ** 2
                    dists[i] = d
                    # print('Dist to center {0}: {1}'.format(i, d))
                cl = sorted(dists, key=dists.get)[0]
                # print('Nearest to center {0}'.format(cl))
                self.members[cl].append(x)
            # Deal with any empty clusters: reassign point at random
            for i, c in self.members.items():
                if c == 0:
                    r = random.choice([j for j in range(1, k+1) if j != k])
                    x = random.choice(self.members[r])
                    self.members[c].append(x)
            # Centroid Update
            new_centers = {}
            for i, c in self.centers.items():
                c = (1 / len(c)) * np.array(self.members[i]).sum(axis=0)
                new_centers[i] = c
            # Calculate MSE
            self.iterations += 1
            print('Iteration:', self.iterations)
            mse = sum([np.linalg.norm(new_centers[new] - self.centers[old]) ** 2
                       for new, old in zip(new_centers, self.centers)])
            print('MSE:', mse)
            delta = abs(mse - self.mse)
            self.mse = mse
            self.centers = new_centers


d = np.random.randn(500, 50)
k = KMeans(4, d)
k.cluster()
