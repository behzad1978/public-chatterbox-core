__author__ = 'behzadbehzadan'

import Pycluster
import scipy

data=[1,2,3,11,12,13,21,22,23]
data = [[d] for d in data]


clusterid = Pycluster.kcluster(data, 5, None, None, 0, 1, 'a', 'e', None)
dist = [[0, 1, 7,],
        [100, 0, 2,],
        [3, 20, 0]]
cluster_id = Pycluster.kmedoids(dist, 2, 1, None)
a = scipy.cluster



print 'stop'