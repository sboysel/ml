"""
Author: Sam Boysel
CS 135 (Tufts University)
Instructor: Anselm Blumer
TA: Mengfei Cao

Usage:
python3 kmeans.py filename.arff

Statistics:
hw5.1.arff
k | Quality
-----------
2 | 343.75
3 | 19.25
4 | 13.75

hw5.2.arff
k | Quality
-----------
2 | 920544.62032085622
3 | 617415.78181604261
4 | 342560.1459521323
5 | 339263.270322
6 | 121592.30018037515
7 | 118714.81786
8 | 118333.96047
9 | 113095.962523
10| 113172.596761
11| 110437.377679
"""

import sys
import random
import numpy as np
from scipy.io.arff import loadarff

def load_data(filename):
    """
    load numeric data from arff file using scipy.io.arff.loadarff
    returns a numpy array
    """
    data = loadarff(open(filename, 'r'))[0]
    return np.array([list(row) for row in data])

def kmeans(k, data, centers = None):
    """
    data = n x m array
    k = integer
    centers = {center_label: center_vector} dict (initialize random centers if
    None)
    """
    obs, feats = data.shape
    center_membership = {}
    # Initial random centers if none given
    if centers == None:
        centers = {m: np.ceil(np.random.uniform(data.min(), data.max(), feats)) for m in range(1, k + 1)}
    for obs in data:
        distances = {} 
        for m in centers.keys():
            distances[m] = np.linalg.norm(obs - centers[m])**2
        m = min(distances, key = distances.get)
        if m in center_membership.keys():
            center_membership[m].append(obs)
        else:
            center_membership[m] = []
            center_membership[m].append(obs)
    # If cluster center is empty, randomly reassign a point from a non-empty center   
    # with length greater than one.
    if len(center_membership) < k:
        nonempty = [i for i in center_membership.keys() if
            len(center_membership[i]) > 1]
        length_zero = [i for i in center_membership.keys() if
                len(center_membership[i]) == 0]
        missing = list(set([i for i in range(1, k + 1)]) - set(nonempty))
        for i in length_zero + missing:
            s = random.choice(nonempty)
            sample = random.choice(center_membership[s])
            center_membership[i] = [sample]
            removearray(center_membership[s], sample)
    # Recalculate centers    
    for m in center_membership.keys():
        newcenter = np.array(center_membership[m]).mean(axis = 0)
        centers[m] = newcenter
    # Quality
    quality = 0    
    for m in center_membership.keys():
        for obs in center_membership[m]:
            dist =  np.linalg.norm(obs - centers[m])**2
            quality = dist + quality

    return centers, center_membership, quality

# For removal of a numpy array from a list by array value
def removearray(l,arr):
    i = 0
    size = len(l)
    while i != size and not np.array_equal(l[i],arr):
        i += 1
    if i != size:
        l.pop(i)
    else:
        raise ValueError('Could not locate array.')

# Tests
def tests_kmeans(data, k_max, tests_max, best_only = False):
    """
    data = load_data('hw5.2.arff')
    krange = range(2, {k_max})
    tests = range(1, {tests_max})
    """
    quality = {}
    for k in range(2, k_max):
        print("k = ", k)
        newc = {}
        for tests in range(1, tests_max + 1):
            for run in range(1, tests_max - 6):
                if run == 1:
                    c, cm, q = kmeans(k, data)
                    quality[k] = [q]
                else:
                    c, cm, q = kmeans(k, data, newc)
                    #print(q)
                    quality[k].append(q)
                newc = c
        for key in quality.keys():
            quality[key] = np.array(quality[key]).min()
    return quality

def main():
    filename = sys.argv[1]
    data = load_data(filename)
    #print('Dataset loaded.')
    k = int(input('Choose k: '))
    newcenters, membership, quality = kmeans(k, data)
    #for key in membership.keys():
    #    print('Center: ', key)
    #    print('Member Points: ')
    #    for point in membership[key]:
    #        print(point)
    print('Sum of Squared Distance from new centers: ', quality)
    while True:
        response = input('Would you like to run another iteration ? (y/n) ')
        if response == 'n':
            response2 = input('Would you like to specify a new value for k? (y/n) ')
            if response2 == 'y':
                k = int(input('Choose k: '))
                newcenters, membership, quality = kmeans(k, data)
            elif response2 == 'n':
                break
        elif response == 'y':
            newcenters, membership, quality = kmeans(k, data, newcenters)
        #for key in membership.keys():
        #    print('Center: ', key)
        #    print('Member Points: ')
        #    for point in membership[key]:
        #        print(point)
        print('Sum of Squared Distance from new centers: ', quality)
        
if __name__ == '__main__':
    main()
