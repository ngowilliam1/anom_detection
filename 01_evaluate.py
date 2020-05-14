import os
import argparse
import time
import pickle
import random
from functions import preprocess, utilities
from modelling import isolation, gan, autoencoder, pid
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from contextlib import redirect_stdout
from sklearn import metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        help = 'Dataset to run.',
        choices=['credit','kdd','mammography','seismic'],
    )
    parser.add_argument(
        'model',
        help = 'Model to run.',
        choices=['PID','IF','AE','GAN','BSP'],
    )
    parser.add_argument(
        '-p','--percentages',
        help='Percent of over anomalies?',
        nargs='+',
        type = utilities.valid_percentage,
        default=[0.05, 0.04, 0.03, 0.02, 0.01, 0.0075, 0.006, 0.005, 0.002, 0.001],
    )
    parser.add_argument(
        '-layer', '--layers',
        help = 'AE Layer Size (Input and then Output).',
        nargs='+',
        type = utilities.valid_strictly_positive_int,
        default=[3, 4, 5, 6],
    )
    parser.add_argument(
        '-latent', '--latent_dim',
        help = 'Latent Dim.',
        nargs='+',
        type = utilities.valid_strictly_positive_int,
        default=[2, 3, 4],
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
        '-ganLR','--gan_learning_rate',
        help='GAN learning rate.',
        type = utilities.valid_percentage,
        default=0.00001,
    )
    parser.add_argument(
        '-ganTE','--gan_total_epochs',
        help='GAN total Epochs.',
        type = utilities.valid_positive_int,
        default=10,
    )
    parser.add_argument(
        '-ganBS','--gan_batch_size',
        help='GAN total Epochs.',
        type = utilities.valid_positive_int,
        default=512,
    )
    parser.add_argument(
        '-aeLR','--ae_learning_rate',
        help='AE learning rate.',
        type = utilities.valid_percentage,
        default=0.00001,
    )
    parser.add_argument(
        '-aeTE','--ae_total_epochs',
        help='AE total epochs.',
        type = utilities.valid_positive_int,
        default=100,
    )
    parser.add_argument(
        '-aeBS','--ae_batch_size',
        help='AE batch size.',
        type = utilities.valid_positive_int,
        default=512,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        choices=[-1,0,1,2,3],
        default=-1,
    )
    parser.add_argument(
        '-catincl', '--cat_incl',
        help='Include categorical variable for Isolation Forest.',
        type=utilities.valid_bool,
        default=True,
    )
    parser.add_argument(
        '-maxsam', '--max_samples',
        help='Max samples for isolation forest.',
        type=utilities.valid_positive_int,
        default=128,
    )
    parser.add_argument(
        '-ntree', '--n_estimators',
        help='Number of trees in IF.',
        type=utilities.valid_positive_int,
        default=100,
    )
    parser.add_argument(
        '-save', '--save_model',
        help='Save Isolation Forest and GAN.',
        type=utilities.valid_bool,
        default=False,
    )
    parser.add_argument(
        '-scale', '--scale',
        help='Type of scaling for data.',
        type=str,
        default="min_max",
        choices=['standard', 'min_max'],
    )
    parser.add_argument(
        '-pid_max_samples', '--pid_max_samples',
        help='PID max samples.',
        default=100,
        type=utilities.valid_positive_int,
    )
    parser.add_argument(
        '-pid_max_depth', '--pid_max_depth',
        help='PID max depth.',
        default=10,
        type=utilities.valid_positive_int,
    )
    parser.add_argument(
        '-pid_n_trees', '--pid_n_trees',
        help='PID N Trees.',
        default=50,
        type=utilities.valid_positive_int,
    )
    args = parser.parse_args()
    ### TENSORFLOW SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    # Make directories if needed (will not overwrite)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    d = []
    timestr = time.strftime("%Y%m%d-%H%M%S")
    for perc in args.percentages:
        random.seed(args.python_seed)
        np.random.seed(args.numpy_seed)
        tf.compat.v1.set_random_seed(args.tensorflow_seed)
        print('% anomaly over dataset:',100*perc,'%')
        # Loading preprocessed data
        if args.cat_incl:
            data_filename = f'../data/preprocessed_{args.dataset}_{perc}_{args.scale}.pkl'
        else:
            data_filename = f'../data/preprocessed_{args.dataset}_NoCat_{perc}_{args.scale}.pkl'
        preprocessed = pickle.load(open(data_filename, 'rb'))

        preprocessed_data = preprocessed[f"pp_{args.scale}"]
        runtime_pp = preprocessed["runtime"]
        validation_labels = preprocessed_data['y_valid']

        if args.model == "IF":
            model = isolation.Isolation_Forest(numpy_seed = args.numpy_seed, max_samples=args.max_samples, n_estimators=args.n_estimators,  include_cat=args.cat_incl)
            runtime_fit = model.fit(x = preprocessed_data['x_train'])
            print("Isolation Forest Training time took:", runtime_fit)
            # Scores represent likelihood that x is a anomaly
            scores = model.evaluate(preprocessed_data['x_valid'])
            aps = metrics.average_precision_score(y_true = preprocessed_data['y_valid'], y_score =scores)
            precisionAt95 = utilities.getPrecisionAtFixedRecall(preprocessed_data['y_valid'], scores, 0.95)
            precisionAt99 = utilities.getPrecisionAtFixedRecall(preprocessed_data['y_valid'], scores, 0.99)
            d.append({'Percent Anomaly': perc, 'Model':args.model, 'Scale Type': args.scale, 'Include Categorical': args.cat_incl
                , 'APS': aps, 'Precision at 95': precisionAt95, 'Precision at 99': precisionAt99
                , 'Runtime': runtime_fit,'Latent Dim': -1, 'AE Layer Size (Input and then Output)': -1
                , 'AE Learning Rate': -1, 'AE Batch Size': -1
                , 'AE Max Epoch': -1, 'GAN Learning Rate': -1, 'GAN Batch Size': -1
                , 'GAN Epoch': -1, 'IF Max Sample': args.max_samples, 'IF N Trees': args.n_estimators
                , 'PID Max Depth': -1, 'PID Trees': -1, 'PID Max Samples': -1
            })
            
            if args.save_model:
                if args.cat_incl:
                    model_filename = f'models/{args.dataset}_{args.model}_maxSam_{args.max_samples}_perc_{perc}_{timestr}.pkl'
                else:
                    model_filename = f'models/{args.dataset}_{args.model}_maxSam_{args.max_samples}_perc_{perc}_NoCat_{timestr}.pkl'
                pickle.dump(model, open(model_filename, 'wb'))
            
        elif args.model == "GAN":
            model = gan.GAN(learning_rate = args.gan_learning_rate, batch_size = args.gan_batch_size, total_epochs = args.gan_total_epochs)
            dictFit = {"save_model":args.save_model,"dataset":args.dataset, "percentage":perc, "timestr":timestr}
            # Scores represent likelihood that x is a anomaly, for gan this is the score at each 2 epoch of the model
            runtime_fit, scores = model.fit(preprocessed_data['x_train'], x_valid = preprocessed_data['x_valid'], dictFit=dictFit)
            print("GAN Training time took:", runtime_fit)
            for idx, score in enumerate(scores):
                aps = metrics.average_precision_score(y_true = preprocessed_data['y_valid'], y_score =score)
                precisionAt95 = utilities.getPrecisionAtFixedRecall(preprocessed_data['y_valid'], score, 0.95)
                precisionAt99 = utilities.getPrecisionAtFixedRecall(preprocessed_data['y_valid'], score, 0.99)
                d.append({'Percent Anomaly': perc, 'Model':args.model, 'Scale Type': args.scale, 'Include Categorical': args.cat_incl
                , 'APS': aps, 'Precision at 95': precisionAt95, 'Precision at 99': precisionAt99
                , 'Runtime': runtime_fit, 'Latent Dim': -1, 'AE Layer Size (Input and then Output)': -1
                , 'AE Learning Rate': -1, 'AE Batch Size': -1
                , 'AE Total Epoch': -1, 'GAN Learning Rate': args.gan_learning_rate, 'GAN Batch Size': args.gan_batch_size
                , 'GAN Epoch': idx*5, 'IF Max Sample': -1
                , 'PID Max Depth': -1, 'PID Trees': -1, 'PID Max Samples': -1
                ,
                })

                
        elif args.model == "AE":
            model = autoencoder.AutoEncoderCollection(args.layers, args.latent_dim, learning_rate = args.ae_learning_rate, batch_size = args.ae_batch_size, total_epochs = args.ae_total_epochs)
            runtime_fit = model.fit(preprocessed_data['x_train'])
            print("AE Collection Training time took:", runtime_fit)
            for key, ae in model.autoEncoders.items():
                # Scores represent likelihood that x is a anomaly
                scores = ae.evaluate(preprocessed_data['x_valid'])
                aps = metrics.average_precision_score(y_true = preprocessed_data['y_valid'], y_score =scores)
                precisionAt95 = utilities.getPrecisionAtFixedRecall(preprocessed_data['y_valid'], scores, 0.95)
                precisionAt99 = utilities.getPrecisionAtFixedRecall(preprocessed_data['y_valid'], scores, 0.99)
                d.append({'Percent Anomaly': perc, 'Model':args.model, 'Scale Type': args.scale, 'Include Categorical': args.cat_incl
                , 'APS': aps, 'Precision at 95': precisionAt95, 'Precision at 99': precisionAt99
                , 'Runtime': runtime_fit,'Latent Dim': ae.latent_dim, 'AE Layer Size (Input and then Output)': ae.layer
                , 'AE Learning Rate': args.ae_learning_rate, 'AE Batch Size': args.ae_batch_size
                , 'AE Max Epoch': args.ae_total_epochs, 'GAN Learning Rate': -1, 'GAN Batch Size': -1
                , 'GAN Epoch': -1, 'IF Max Sample': -1
                , 'PID Max Depth': -1, 'PID Trees': -1, 'PID Max Samples': -1
                })
                
        elif args.model == "PID":
            # Extra preprocessing since PID cannot handle no entropy columns:
            colsToDelete = []
            for axis in range(preprocessed_data['x_train'].shape[1]):
                val = np.unique(preprocessed_data['x_train'][:,axis])
                if len(val) <= 1:
                    colsToDelete.append(axis)

            preprocessed_data['x_train'] = np.delete(preprocessed_data['x_train'], colsToDelete, 1)
            preprocessed_data['x_valid'] = np.delete(preprocessed_data['x_valid'], colsToDelete, 1)
            
            kwargs = {'max_depth': args.pid_max_depth, 'n_trees':args.pid_n_trees,  'max_samples': args.pid_max_samples, 'max_buckets': 3, 'epsilon': 0.1, 'sample_axis': 1, 
                      'threshold': 0}
            model = pid.pidForest(**kwargs)
            runtime_fit = model.fit(preprocessed_data['x_train'])
            # Scores represent likelihood that x is a anomaly
            scores = model.evaluate(preprocessed_data['x_valid'])
            aps = metrics.average_precision_score(y_true = preprocessed_data['y_valid'], y_score =scores)
            precisionAt95 = utilities.getPrecisionAtFixedRecall(preprocessed_data['y_valid'], scores, 0.95)
            precisionAt99 = utilities.getPrecisionAtFixedRecall(preprocessed_data['y_valid'], scores, 0.99)
            d.append({'Percent Anomaly': perc, 'Model':args.model, 'Scale Type': args.scale, 'Include Categorical': args.cat_incl
            , 'APS': aps, 'Precision at 95': precisionAt95, 'Precision at 99': precisionAt99
            , 'Runtime': runtime_fit, 'Latent Dim': -1, 'AE Layer Size (Input and then Output)': -1
            , 'AE Learning Rate': -1, 'AE Batch Size': -1
            , 'AE Total Epoch': -1, 'GAN Learning Rate': -1, 'GAN Batch Size': -1
            , 'GAN Epoch': -1, 'IF Max Sample': -1
            , 'PID Max Depth': args.pid_max_depth, 'PID Trees': args.pid_n_trees, 'PID Max Samples': args.pid_max_samples
            ,
            })
            if args.save_model:
                if args.cat_incl:
                    model_filename = f'models/{args.dataset}_{args.model}_maxSam_{args.pid_max_samples}_maxDepth_{args.pid_max_depth}_nTrees_{args.pid_n_trees}_perc_{perc}_{timestr}.pkl'
                else:
                    model_filename = f'models/{args.dataset}_{args.model}_maxSam_{args.pid_max_samples}_maxDepth_{args.pid_max_depth}_nTrees_{args.pid_n_trees}_perc_{perc}_NoCat_{timestr}.pkl'
                pickle.dump(model, open(model_filename, 'wb'))
            
    d = pd.DataFrame(d)
    d.to_csv(f'results/{args.dataset}_{args.model}_' + timestr + '.csv', index=False)

    hyperparam_filename = f'results/{args.dataset}_{args.model}_{timestr}.txt'
    if args.model == 'GAN':
        with open(hyperparam_filename,"w") as f:
            f.write(f'Dataset: {args.dataset}\n')
            f.write(f'Model: {args.model}\n')
            f.write(f'Percentages: {args.percentages}\n')
            f.write(f'TF_Seed: {args.tensorflow_seed}\n')
            f.write(f'NP_Seed: {args.numpy_seed}\n')
            f.write(f'Py_Seed: {args.python_seed}\n')
            f.write(f'GAN_Learning_Rate: {args.gan_learning_rate}\n')
            f.write(f'GAN_Batch_size: {args.gan_batch_size}\n')
            f.write(f'GAN_Total_epoch: {args.gan_total_epochs}\n')
            with redirect_stdout(f):
                f.write('Discriminator:\n')
                model.discriminator.summary()
                f.write('\nGenerator:\n')
                model.generator.summary()

            f.close() 