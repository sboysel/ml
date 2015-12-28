"""
EM Clustering
Section 13.3: Data Mining and Analysis (Zaki & Meira, 2014)

Resources:
http://www.mathworks.com/help/stats/gmdistribution.fit.html?s_tid=gn_loc_drop
"""

import numpy as np
from scipy.stats import multivariate_normal as MVN


class EMCluster:

    def __init__(self, data, k, epsilon=0.001):
        self.data = data
        self.k = k
        self.dims = data.shape
        self.n = data.shape[0]
        self.d = data.shape[1]
        self.epsilon = epsilon
        self.mu = {i: [np.random.uniform(np.min(data),
                                         np.max(data),
                                         data.shape[1])]
                   for i in range(k)}
        self.sigma = {i: [np.eye(self.dims[1])] for i in range(k)}
        self.p = {i: [(1 / k)] for i in range(k)}
        self.t = 0
        self.w = {i: [] for i in range(k)}
        self.is_clustered = False

    def get_clusters(self):
        """Returns the latest set of clusters after convergence.
        """
        if not self.is_clustered:
            self.cluster()
        return({c: mu[self.t] for c, mu in self.mu.items()})

    def error(self):
        """Calculate the change in cluster centers from the last iteration.
        Used to determine convergence
        """
        if self.t == 0:
            return(np.nan)
        elif self.t == 1:
            sse = sum([np.linalg.norm(v[0]) ** 2 for k, v in self.mu.items()])
            return(sse)
        else:
            sse = sum([np.linalg.norm(v[-1] - v[-2]) ** 2
                       for k, v in self.mu.items()])
            return(sse)

    def __posterior_prob(self, t, i, j):
        """Calculate the posterior probability of cluster i given data j.  For use
        in expectation step in EMCluster.cluster() method.

        Parameters
        ----------

        self : EMCluster object

        t : int, Loop iteration.

        i : int, Cluster identifier.

        j : int, Observation x_j.
        """
        num = MVN.pdf(self.data[j,:],
                      self.mu[i][t - 1],
                      self.sigma[i][t - 1]) * self.p[i][t - 1]
        denom = sum([MVN.pdf(self.data[j,:],
                             self.mu[l][t - 1],
                             self.sigma[l][t - 1]) * self.p[l][t - 1]
                     for l in range(self.k)])
        post = (num / denom)
        assert 0 <= post <= 1
        return(post)

    def cluster(self, sigma_diag=True, regularize=False):
        """Cluster input data using Expectation Maximization

        Parameters
        ----------

        sigma_diag : bool, default=True.  If True, use only the diagonal for
            each variance-covariance matrix (i.e. variance components only).

        regularize : bool, default=False.  If True, add a small positive number
            to the diagonal of each variance-covariance matrix.
        """
        self.t = 0
        while self.error() > self.epsilon or self.t == 0:
            self.t += 1
            # Expectation Step
            for i in range(self.k):
                W = []
                for j in range(self.dims[0]):
                    w = self.__posterior_prob(self.t, i, j)
                    W.append(w)
                self.w[i].insert(self.t, W)
            # Maximization Step
            for i in range(self.k):
                mu_num = sum([self.w[i][self.t - 1][j] * self.data[j,:]
                              for j in range(self.dims[0])])
                sum_wij = sum([self.w[i][self.t - 1][j]
                               for j in range(self.dims[0])])
                mu = (mu_num / sum_wij)
                sigma_num = sum([self.w[i][self.t - 1][j] *
                                 np.outer((self.data[j,:] - mu),
                                          (self.data[j,:] - mu))
                                 for j in range(self.dims[0])])
                sigma = (sigma_num / sum_wij)
                # Methods to avoid ill-conditioned variance-covariance matrices
                if sigma_diag:
                    sigma = np.diag(np.diag(sigma))
                if regularize:
                    reg = np.diag(np.diag(np.random.uniform(low=0,
                                                            high=0.01,
                                                            size=sigma.shape)))
                    sigma = sigma + reg
                p = (sum_wij / self.dims[0])
                self.mu[i].insert(self.t, mu)
                self.sigma[i].insert(self.t, sigma)
                self.p[i].insert(self.t, p)
        self.is_clustered = True
