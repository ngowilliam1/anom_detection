import os
import argparse
import time
import pickle
import random
from functions import preprocess, utilities
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        help = 'Dataset to run.',
        choices=['credit','kdd','mammography'],
    )
    parser.add_argument(
        '-p','--percentages',
        help='Percent of over anomalies?',
        nargs='+',
        type = utilities.valid_percentage,
        default=[0.05, 0.04, 0.03, 0.02, 0.01, 0.0075, 0.006, 0.005, 0.002, 0.001],
    )
    parser.add_argument(
        '-cat', '--cat_vars',
        help = 'Names of Categorical Variables.',
        nargs='+',
        type = str,
        default=['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login'],
    )
    parser.add_argument(
        '-tfs', '--tensorflow_seed',
        help = 'Seed for GAN.',
        type = utilities.valid_positive_int,
        default= 123,
    )
    parser.add_argument(
        '-nps', '--numpy_seed',
        help = 'Seed for numpy preprocessing.',
        type = utilities.valid_positive_int,
        default= 123,
    )
    parser.add_argument(
        '-pys', '--python_seed',
        help = 'Seed for python Random.',
        type = utilities.valid_positive_int,
        default= 123,
    )
    parser.add_argument(
        '-catincl', '--cat_incl',
        help='Include categorical variable for Isolation Forest.',
        type=utilities.valid_bool,
        default=True,
    )
    parser.add_argument(
        '-scale', '--scale',
        help='Type of scaling for data.',
        type=str,
        default="min_max",
        choices=['standard', 'min_max'],
    )
    args = parser.parse_args()
    print(args.cat_vars)

    # Generating data for each anomaly percentage
    for perc in args.percentages:
        random.seed(args.python_seed)
        np.random.seed(args.numpy_seed)
        tf.compat.v1.set_random_seed(args.tensorflow_seed)
        print('% anomaly over dataset:',100*perc,'%')
        # Preprocessing
        preprocessed = preprocess.data_proc(dataset_name=args.dataset,
                                            perc=perc,
                                            scaling=args.scale,
                                            cat_vars=args.cat_vars,
                                            cat_incl = args.cat_incl
                                            )
        if args.cat_incl:
            filename = f'../data/preprocessed_{args.dataset}_{perc}_{args.scale}.pkl'
        else:
             filename = f'../data/preprocessed_{args.dataset}_NoCat_{perc}_{args.scale}.pkl'
        # Save preprossed data as pickle file
        Path("../data").mkdir(parents=True, exist_ok=True)
        pickle.dump(preprocessed, open(filename, 'wb'))