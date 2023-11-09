#!/usr/bin/env python
# coding: utf-8

# # Fed-BioMed Researcher to train miwae with adni dataset

# ## Start the network
# Before running this notebook, start the network with `./scripts/fedbiomed_run network`
#
## Start the network and setting the node up
# Before running this notebook, you shoud start the network from fedbiomed-network, as detailed in https://gitlab.inria.fr/fedbiomed/fedbiomed-network
# Therefore, it is necessary to previously configure a node:
# 1. `./scripts/fedbiomed_run node add`
#   * Select option 1 to add a csv file to the node
#   * Choose the name, tags and description of the dataset
#     * use `#test_data`` for the tags
#   * Pick the .csv file from your PC (here: pseudo_adni_mod.csv)
#   * Data must have been added
# 2. Check that your data has been added by executing `./scripts/fedbiomed_run node list`
# 3. Run the node using `./scripts/fedbiomed_run node start`. Wait until you get `Starting task manager`. it means you are online.


# ## Create an experiment to train a model on the data found



# Declare a torch training plan MyTrainingPlan class to send for training on the node
import numpy as np
import torch
import pandas as pd
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, mean_squared_error, hinge_loss

from func_miwae_traumabase import databases, databases_pred, generate_save_plots_prediction,\
    recover_data_prediction, save_results_prediction, save_model, load_and_predict_data, save_results_load_and_predict
from class_miwae_traumabase import FedMeanStdTrainingPlan, SGDRegressorTraumabaseTrainingPlan, FedLogisticRegTraumabase

from fedbiomed.researcher.experiment import Experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline fed standardization and hemo shock prediction, Traumabase')
    parser.add_argument('--method', metavar='-m', type=str, default='FedAvg', choices = ['FedAvg', 'FedProx', 'FedProx_loc', 'Scaffold', 'Local', 'Centralized'],
                        help='Methods for the running experiment')
    parser.add_argument('--regressor', metavar='-r', type=str, default='linear', choices = ['linear', 'logistic'],
                        help='Methods for the running experiment')
    parser.add_argument('--task', metavar='-ts', type=str, default='prediction', choices = ['prediction','load_and_predict'],
                        help='Task to be performed with the pipeline')
    parser.add_argument('--Test_id', metavar='-tid', type=int, default=4,
                        help='Id of the Test dataset (between 1 and 4)')
    parser.add_argument('--tags', metavar='-t', type=str, default='traumabase_pred', help='Dataset tags')
    parser.add_argument('--Rounds', metavar='-r', type=int, default=100,
                        help='Number of rounds for imputation')
    parser.add_argument('--Epochs', metavar='-e', type=int, default=5,
                        help='Number of epochs for imputation')
    parser.add_argument('--data_folder', metavar='-d', type=str, default='data/',
                        help='Datasets folder')
    parser.add_argument('--root_data_folder', metavar='-rdf', type=str, default=None, choices=['fedbiomed','home'],
                        help='Root directory for data')
    parser.add_argument('--result_folder', metavar='-rf', type=str, default='results', 
                        help='Folder cotaining the results csv')
    parser.add_argument('--batch_size', metavar='-bs', type=int, default=48,
                        help='Batch size')
    parser.add_argument('--standardize', metavar='-std', default=True, action=argparse.BooleanOptionalAction,
                        help='Standardize data for regression')
    parser.add_argument('--do_fed_std', metavar='-fstd', default=True, action=argparse.BooleanOptionalAction,
                        help='Recover federated mean and std')
    parser.add_argument('--do_figures', metavar='-fig', default=True, action=argparse.BooleanOptionalAction,
                        help='Generate and save figures during local training')
    parser.add_argument('--num_samples', metavar='-ns', type=int, default=30,
                        help='Number of multiple imputations (if kind == both or kind == multiple)')

    args = parser.parse_args()

    method = args.method
    task = args.task
    idx_Test_data = int(args.Test_id)
    tags = args.tags
    data_folder = args.data_folder
    root_dir = args.root_data_folder
    regressor = args.regressor
    result_folder = args.result_folder
    num_samples = args.num_samples
    w0=5/6
    w1=1/6

    target_col = ['choc_hemo']

    ###########################################################
    # Recover data size                                       #
    ###########################################################

    from fedbiomed.researcher.requests import Requests
    req = Requests()
    xx = req.list()
    dataset_size = [xx[i][0]['shape'][1] for i in xx]
    min_n_samples = min([xx[i][0]['shape'][0] for i in xx])
    assert min(dataset_size)==max(dataset_size)
    data_size = dataset_size[0]

    num_covariates = 13

    regressors_col = ['fracas_du_bassin_-1.0', 'fracas_du_bassin_0.0', 'fracas_du_bassin_1.0', 
                    'catecholamines_-1.0', 'catecholamines_0.0', 'catecholamines_1.0', 
                    'intubation_orotracheale_smur_-1.0', 'intubation_orotracheale_smur_0.0', 'intubation_orotracheale_smur_1.0', 
                    'sexe', 'expansion_volemique', 'penetrant', 'age', 'pression_arterielle_systolique_PAS_minimum', 
                    'pression_arterielle_diastolique_PAD_minimum', 'frequence_cardiaque_FC_maximum', 'hemocue_initial']

    #Number of partecipating clients
    N_cl = len(dataset_size)

    ###########################################################
    # Recover full dataset and test dataset for testing phase #
    ###########################################################

    idx_clients=[*range(1,N_cl+2)]
    idx_clients.remove(idx_Test_data)

    Clients_data, data_test = databases_pred(data_folder=data_folder,idx_clients=idx_clients,
                                            root_dir=root_dir,idx_Test_data=idx_Test_data, imputed = True)

    ###########################################################
    # Recover global mean and std in a federated manner       #
    ###########################################################

    fed_mean, fed_std = None, None

    if args.standardize:
        if method not in ['FedProx_loc','Local','Centralized']:
            if args.do_fed_std ==  True:

                from fedbiomed.researcher.aggregators.fedstandard import FedStandard

                # NOTE: we need to perform only 1 round of 1 epoch to recover global mean and std
                model_args = {'n_features': data_size, 'n_cov': num_covariates}

                training_args = {
                    'loader_args': {
                        'batch_size': 48,
                        'drop_last': True
                    },
                    'optimizer_args': {
                        'lr': 0
                    }, 
                    'log_interval' : 1,
                    'num_updates': 1, 
                    'dry_run': False,  
                }

                fed_mean_std = Experiment(tags=tags,
                                model_args=model_args,
                                training_plan_class=FedMeanStdTrainingPlan,
                                training_args=training_args,
                                round_limit=1,
                                aggregator=FedStandard(),
                                node_selection_strategy=None)

                fed_mean_std.run()
                fed_mean = fed_mean_std.aggregated_params()[0]['params']['mean']
                fed_std = fed_mean_std.aggregated_params()[0]['params']['std']

            else:
                npzfile = np.load(args.result_folder+'/clients_imputed/'+method+'_mean_std.npz')
                fed_mean, fed_std = npzfile['mean'], npzfile['std']

    ###########################################################
    #Define the hyperparameters for SGDregressor              #
    ###########################################################

    tol = 1e-5
    eta0 = 0.05
    n_epochs=args.Epochs
    batch_size = args.batch_size
    num_updates = int(np.ceil(min_n_samples/batch_size)*n_epochs)
    rounds = args.Rounds

    ###########################################################
    #Define the federated SGDregressor model                  #
    ###########################################################
    if method not in ['Local','Centralized']:
        model_args = {'n_features': len(regressors_col), 'n_cov': num_covariates-1, 'use_gpu': True, 
                    'regressors_col':regressors_col, 'target_col': target_col, 'tol': tol,'eta0': eta0,'random_state':1234}
        if args.standardize:
            print('standardization added')
            standardization = {} if method == 'FedProx_loc' else {'fed_mean':fed_mean.tolist(),'fed_std':fed_std.tolist()}
            model_args.update(standardization=standardization)

        training_args = {
            'loader_args': {
                'batch_size': batch_size,
                'drop_last': True
            },
            'num_updates': num_updates, 'dry_run': False}

        if regressor == 'logistic':
            model_args.update(n_classes = 2)
            training_plan = FedLogisticRegTraumabase
        elif regressor == 'linear':
            training_plan = SGDRegressorTraumabaseTrainingPlan

        ###########################################################
        #Declare the experiment                                   #
        ###########################################################

        from fedbiomed.researcher.aggregators.fedavg import FedAverage
        from fedbiomed.researcher.aggregators.scaffold import Scaffold

        aggregator = Scaffold() if method == 'Scaffold' else FedAverage()

        if 'fedprox_mu' in training_args:
            del training_args['fedprox_mu']

        exp = Experiment(tags=tags,
                        model_args=model_args,
                        training_plan_class=training_plan,
                        training_args=training_args,
                        round_limit=rounds,
                        aggregator=aggregator,
                        node_selection_strategy=None)

        exp.run_once()

    if task != 'load_and_predict':
        if method not in ['Local','Centralized']:
            if 'FedProx' in method:
                # Starting from the second round, FedProx is used with mu=0.1
                # We first update the training args
                training_args.update(fedprox_mu = 0.1)

                # Then update training args in the experiment
                exp.set_training_args(training_args)

            exp.run()

        ###########################################################
        #Local model                                              #
        ###########################################################    
        elif method == 'Local':

            #TO BE FILLED

            if args.do_figures==True:
                Loss_cls = [[] for _ in range(N_cl)]
                Accuracy_cls = [[] for _ in range(N_cl)]
                MSE_cls = [[] for _ in range(N_cl)]

            # Recall all hyperparameters
            n_epochs_local = n_epochs*rounds
            n_epochs_centralized = n_epochs*rounds*N_cl

            # .................

            Coeff_loc = []
            Intercept_loc = []

            #for cls in range(N_cl):
                
                # .................

            #    for ep in range(1,n_epochs_local):
                    
                    # .................

                # Append updated coeff, intercept

        elif method == 'Centralized':
            # Centralized training
            #if args.do_figures==True:
            #    Loss_tot = []
            #    Accuracy_tot = []
            #    MSE_tot = []

            Data_tot = pd.concat(Clients_data, ignore_index=True)

            # Training loop

            #for ep in range(1,n_epochs_centralized):

                # .................

        ###########################################################
        #Testing phase (imputation)                               #
        ###########################################################

        if task == 'prediction':
            X_test = data_test[regressors_col].values
            y_test = data_test[target_col].values.astype(int)
            
            if args.standardize:
                xfull_global_std, xfull_local_std = recover_data_prediction(X_test, num_covariates-1, fed_mean, fed_std)
                X_test=xfull_global_std

            # we create here several instances of SGDRegressor using same sklearn arguments
            # we have used for Federated Learning training
            model_pred = exp.training_plan().model()
            regressor_args = {key: model_args[key] for key in model_args.keys() if key in model_pred.get_params().keys()}

            testing_error = []
            Validation = []
            Losses = []
            Accuracies = []

            for i in range(rounds):
                model_pred.coef_ = exp.aggregated_params()[i]['params']['coef_'].copy()
                model_pred.intercept_ = exp.aggregated_params()[i]['params']['intercept_'].copy()
                y_pred = model_pred.predict(X_test).astype(int)
                mse = np.mean((y_pred - y_test)**2)
                testing_error.append(mse)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                validation_err = (w0*fn+w1*fp)/(fn+fp)
                Validation.append(validation_err)
                prediction = model_pred.decision_function(X_test)
                Losses.append(hinge_loss(y_test, prediction))
                Accuracies.append(model_pred.score(X_test,y_test))


            model_pred.coef_ = exp.aggregated_params()[rounds - 1]['params']['coef_'].copy()
            model_pred.intercept_ = exp.aggregated_params()[rounds - 1]['params']['intercept_'].copy() 

            save_model(result_folder,method,regressor,model_pred.coef_,model_pred.intercept_)

            y_pred = model_pred.predict(X_test).astype(int)
            #if regressor == 'logistic':
            #    y_predict_prob = model_pred.predict_proba(X_test)
            #    y_predict_prob_class_1 = y_predict_prob[:,1]
            #    y_pred = [1 if prob > 0.45 else 0 for prob in y_predict_prob_class_1]

            conf_matr = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            accuracy = accuracy_score(y_test, y_pred)
            F1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred,zero_division=0)
            mse = mean_squared_error(y_test, y_pred)

            validation_err = (w0*fn+w1*fp)/(fn+fp)    

            coefs = model_pred.coef_
            
            # feat_names = model_pred.feature_names_in_ if regressor == 'logistic' else None
            feat_names = None

            save_results_prediction(result_folder, method, regressor, n_epochs, rounds, F1, 
                                    precision, mse, accuracy, conf_matr, validation_err,
                                    coefs, feat_names)

            if args.do_figures==True:
                generate_save_plots_prediction(result_folder,testing_error,Validation,Losses,Accuracies,conf_matr,method,regressor)

    elif task == 'load_and_predict':
        # Y_pred = []
        N = data_test.shape[0]
        Y_pred_n = [[] for n in range(N)]

        # multiple
        for i in range(num_samples):
            model_pred = exp.training_plan().model()
            model_pred.coef_ = torch.load(f'{result_folder}/{method}_{regressor}_trained_model_coef')
            model_pred.intercept_ = torch.load(f'{result_folder}/{method}_{regressor}_trained_model_intercept')
            data_test_mul = load_and_predict_data(sam = i,data_folder=data_folder,idx_Test_data=idx_Test_data,root_dir=root_dir)
            X_test = data_test_mul[regressors_col].values
            y_test = data_test_mul[target_col].values.astype(int)
            
            if args.standardize:
                xfull_global_std, xfull_local_std = recover_data_prediction(X_test, num_covariates-1, fed_mean, fed_std)
                X_test=xfull_global_std

            y_pred = model_pred.predict(X_test).astype(int)
            for n in range(N):
                Y_pred_n[n].append(y_pred[n])
            # Y_pred.append(y_pred)

        # single
        model_pred = exp.training_plan().model()
        model_pred.coef_ = torch.load(f'{result_folder}/{method}_{regressor}_trained_model_coef')
        model_pred.intercept_ = torch.load(f'{result_folder}/{method}_{regressor}_trained_model_intercept')
        Clients_data, data_test_sing = databases_pred(data_folder=data_folder,idx_clients=idx_clients,
                                            root_dir=root_dir,idx_Test_data=idx_Test_data, imputed = True)
        X_test = data_test_sing[regressors_col].values
        y_test = data_test_sing[target_col].values.astype(int)
        
        if args.standardize:
            xfull_global_std, xfull_local_std = recover_data_prediction(X_test, num_covariates-1, fed_mean, fed_std)
            X_test=xfull_global_std

        y_pred = model_pred.predict(X_test).astype(int)
        for n in range(N):
            Y_pred_n[n].append(y_pred[n])

        # Y_pred_n = []

        # Y_pred_n = [Y_pred[sam][n] for sam in range(num_samples) for n in range(len(y_pred))]
        Prediction_df = pd.DataFrame(columns = ['Strong_pred', 'Prob_pred_0', 'Prob_pred_1', 'True_label', 'Correct'])

        Y_pred_strong = []
        Y_pred_prob = []
        for n in range(N):
            samples = len(Y_pred_n[n])
            print(samples,list(set(Y_pred_n[n])))
            count_0 = Y_pred_n[n].count(0)/float(samples)
            count_1 = Y_pred_n[n].count(1)/float(samples)
            vote = 0 if count_0>count_1 else 1
            Y_pred_strong.append(vote)
            Y_pred_prob.append([count_0,count_1])
            Prediction_df = Prediction_df._append({'Strong_pred': vote, 'Prob_pred_0': count_0, \
                                                  'Prob_pred_1': count_1, 'True_label': y_test[n], \
                                                    'Correct': True if vote == y_test[n] else False},ignore_index=True)

        conf_matr = confusion_matrix(y_test, np.array(Y_pred_strong))
        tn, fp, fn, tp = confusion_matrix(y_test, Y_pred_strong).ravel()
        accuracy = accuracy_score(y_test, Y_pred_strong)
        F1 = f1_score(y_test, Y_pred_strong)
        precision = precision_score(y_test, Y_pred_strong,zero_division=0)
        mse = mean_squared_error(y_test, Y_pred_strong)
        validation_err = (w0*fn+w1*fp)/(fn+fp)       
  
        save_results_load_and_predict(result_folder,Prediction_df)

        # print(Y_pred_prob)
        # print(Y_pred_strong)
        print(conf_matr)
        print(accuracy)
        print(validation_err)