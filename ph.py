#!/usr/bin/env python3

import numpy as np
from moabb.datasets import Cho2017
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from tqdm import tqdm

# setup which dataset to consider from MOABB
dataset = Cho2017()
paradigm = MotorImagery()
paradigm_name = 'MI'

# which subjects to consider
subject_source = 1
subject_target = 3
ncovs_target_train = 20

# set the weights for each class in the dataset
weights_classes = {}
weights_classes['left_hand'] = 1
weights_classes['right_hand'] = 1

# get the data for the source and target subjects
data_source = {}
data_target = {}
X, labels, meta = paradigm.get_data(dataset, subjects=[subject_source])
data_source['covs'] = Covariances(estimator='lwf').fit_transform(X)
data_source['labels'] = labels
print(X, labels, meta)
X, labels, meta = paradigm.get_data(dataset, subjects=[subject_target])
data_target['covs'] = Covariances(estimator='lwf').fit_transform(X)
data_target['labels'] = labels
print(X, labels, meta)



	