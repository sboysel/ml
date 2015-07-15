"""
AdaBoost Implemetation (following figure 17.2, Alpaydin, 2005)
Author: Sam Boysel
CS135
Instructor: Anselm Blumer
TA: Mengfei Cao

Usage:
python3 adaboost.py n_learners train.arff test.arff

Results:
Learners    Average Error on test set   Running Time
1           0.241                       0:00:00.078914
5           0.223                       0:00:00.359363
10          0.219                       0:00:00.704969
100         0.207                       0:00:06.998875
1000        0.199                       0:01:09.546881
5000        0.195                       0:05:49.818688

The weights [log(1 / beta)] are printed along with the rest of the program
output.  I did not include them in this comment given it would have taken too
much space.  My guess as to why the error rate only declines marginally wit the
addition of more learners is that my learner training function does a good job
of finding the error minimizing attribute to split on (perhaps this results in 
learners that are too strong for AdaBoost?).  This obviously comes at the
expense of running time.

Note: one thing I hard coded was the step width (=4) for finding the optimal
threshold for each learner over the range of each attribute (i.e. learner
trains over 4 bins in each attribute and chooses the threshold that minimizes
error).
"""

import argparse
import pprint
import numpy as np
from scipy.io.arff import loadarff
from datetime import datetime

parser = argparse.ArgumentParser(description = 'Parse command line arguments.')

parser.add_argument('learners', type = int, help = 'Integer number of base learners to use.')
parser.add_argument('train', type = str, help = 'Train dataset filename.  Must be valid .arff')
parser.add_argument('test', type = str, help = 'Test dataset filename.  Must be valid .arff')

args = parser.parse_args()

"""
INITIALIZATION/UTILITY FUNCTIONS
"""

def load_data(filename):
    """
    returns an array of floats givent the specified filename.
    requires scipy.io.arff.loadarff
    """
    raw = loadarff(filename)[0]
    return np.array([[float(i) for i in row] for row in raw])

def make_ensemble(L, init_threshold = 0.): 
    """
    Initialize ensemble array of untrained learners.
    """
    ensemble = {}
    for j in range(L):
        ensemble[j] = {'attribute': None, 'threshold': init_threshold, 
                'direction': '', 'error': np.inf, 'beta': 0, 'weight': 0}
    return ensemble

"""
TRAINING FUNCTIONS
"""

def init_probs(train_data):
    """
    Initialize probability of drawing instance pair j to 1/N
    where N is the number of oberservations in the training set.
    """
    return np.array([1/train_data.shape[0]] * train_data.shape[0])

def random_draw(train_data, input_probs, sample_size = 1.0):
    """
    Input is an array of probabilities of drawing a specific example
    Output is a random draw from that array according to where a random
    uniform draw lies within the CDF of the input array.
    """
    cum_probs = np.cumsum(input_probs)
    draws = np.random.uniform(size = int(train_data.shape[0] * sample_size))
    sample_indices = []
    for i, d in enumerate(draws):
        for j in range(len(cum_probs)):
            if j > 0 and (d <= cum_probs[j] and d > cum_probs[j - 1]):
                sample_indices.append(j)
            elif j == 0 and d <= cum_probs[j]:
                sample_indices.append(j)
    # return sample, sample_indices, sample_labels
    return train_data[sample_indices,:-1], sample_indices, train_data[sample_indices,-1]

def error(labels, preds, probs = []):
    """
    For learner j, calculate the probability weighted error, epsilon_j.
    """
    e = 0.
    if len(probs) == 0:
        probs = np.ones(len(labels))
    for r, y, p in zip(labels, preds, probs):
        if r != y:
            e += 1.*(p)
    return e

def eval_learner(train_data, labels, learner_index, ensemble_array, probs = [], attr =
    'stored_value', direction = 'stored_value'):
    """
    Evaluate learner j on training data. Returns preductions and error. 
    """
    threshold = ensemble_array[learner_index]['threshold']
    if attr == 'stored_value':
        attr = ensemble_array[learner_index]['attribute']
    if direction == 'stored_value':    
        direction = ensemble_array[learner_index]['direction']
    preds = []
    for obs in train_data[:,attr]:
        if direction == 'lt':
            if obs <= threshold:
                preds.append(1.)
            else:
                preds.append(-1.)
        if direction == 'gt':
            if obs >= threshold:
                preds.append(1.)
            else:
                preds.append(-1.)
    e = error(labels, preds, probs = probs)
    return preds, e

def train_learner(train_data, labels, learner_index, ensemble_array):
    """
    Trains learner j (a decision stump) that splits attribute that gives the lowest
    classification error on the training set.  Stores all updates to learner in
    ensemble_array (threshold, direction, error).
    """
    for attr in range(train_data.shape[1] - 1):  
        feats = train_data[:,attr]
        a, b = feats.min(), feats.max()
        step = feats.ptp()/4
        #step = feats.ptp()/float(len(feats))
        for s in np.arange(a, b + 1, step):
            for direc in ['lt', 'gt']:
                old_threshold = ensemble_array[learner_index]['threshold']
                old_direction = ensemble_array[learner_index]['direction']
                old_attribute = ensemble_array[learner_index]['attribute']
                ensemble_array[learner_index]['threshold'] = s
                ensemble_array[learner_index]['direction'] = direc
                ensemble_array[learner_index]['attribute'] = attr
                preds, error = eval_learner(train_data, labels, learner_index,
                        ensemble_array, probs = [], attr = attr, direction = direc)
                if error < ensemble_array[learner_index]['error']:
                    ensemble_array[learner_index]['error'] = error
                else:
                    ensemble_array[learner_index]['threshold'] = old_threshold
                    ensemble_array[learner_index]['direction'] = old_direction
                    ensemble_array[learner_index]['attribute'] = old_attribute

def update_probs(beta_j, probs, sample, sample_index, preds, labels):
    """
    Update probability set using beta_j = error/(1 - error) for correct
    instances.
    """
    running_list = []
    for obs, i, y, r in zip(sample, sample_index, preds, labels):
        if i in running_list:
            continue
        else:
            running_list.append(i)
            if y == r:
                probs[i] = beta_j * probs[i]
    return probs

def normalize_probs(probs):
    return  np.array(probs) / np.array(probs).sum()

"""
TESTING FUNCTIONS
"""

### TEST ###
def testing(test_data, ensemble, probs):
    """
    For each observation in the training set, calculate a weighted vote for
    each learner using its trained attribute and stores beta_j.  The prediction
    on the observation is then the weighted sum.
    """
    class_outputs, final_preds = [], []
    error = 0.
    for obs in test_data:
        votes = []
        label = obs[-1]
        for learner in ensemble.keys():
            attr = ensemble[learner]['attribute']
            beta_j = ensemble[learner]['beta']
            threshold = ensemble[learner]['threshold']
            direc = ensemble[learner]['direction']
            # Simplified eval_learner() for scalars
            elem = obs[attr]
            if direc == 'lt':
                if elem <= threshold:
                    d_ij = 1.
                else:
                    d_ij = -1.
            if direc == 'gt':
                if elem >= threshold:
                    d_ij = 1.
                else:
                    d_ij = -1.
            w = np.log(1 / beta_j) 
            votes.append(w * d_ij)
        y_i = sum(votes)
        class_outputs.append(y_i)
        if y_i > 0:
            final_preds.append(1)
            if label == -1.:
                error += 1
        else:
            final_preds.append(-1)
            if label == 1.:
                error += 1
    print('Learners Trained:', args.learners)
    print('Learners Used:', len(ensemble))
    print('Error:', error / test_data.shape[0])
    return class_outputs, final_preds 

"""
MAIN
"""

def main():
    """
    Wrapper function to build AdaBoost algorithm.
    """
    start = datetime.now()
    train = load_data(args.train)
    test = load_data(args.test)
    ensemble = make_ensemble(args.learners, init_threshold = 0.)
    probs = init_probs(train)
    for j in range(args.learners):
        sample, sample_indices, labels = random_draw(train, probs)
        print('Training Learner {}'.format(j))
        train_learner(sample, labels, j, ensemble)
        preds, error = eval_learner(sample, labels, j, ensemble, probs = probs)
        print('Error:', error)
        if error > 0.5:
            print('Removing...')
            ensemble.pop(j, None)
            continue
        else:
            beta = error/(1. - error)
            ensemble[j]['beta'] = beta
            ensemble[j]['weight'] = np.log(1 / beta)
        probs = update_probs(beta, probs, sample, sample_indices, preds, labels)
        probs = normalize_probs(probs)
    pprint.pprint(ensemble)
    class_outputs, final_predictions = testing(test, ensemble, probs)
    print('Running Time:', datetime.now() - start)

if __name__ == '__main__':
    main()
