"""
Testing emclustering
"""

import numpy as np
from emclustering import EMCluster


X = np.random.randn(100, 10)
K = EMCluster(data=X, k=4)
K.get_clusters()
