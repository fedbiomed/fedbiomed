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
import argparse
import sys

from func_miwae_traumabase import miwae_loss, encoder_decoder_iota_opt, recover_data, testing_func, save_results_imputation, \
    databases, generate_save_plots, create_save_data_prediction, testing_func_mul
from class_miwae_traumabase import FedMeanStdTrainingPlan, MIWAETrainingPlan

from fedbiomed.researcher.experiment import Experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline fed standardization and imputation, Traumabase')
    parser.add_argument('--method', metavar='-m', type=str, default='FedAvg', choices = ['FedAvg', 'FedProx', 'FedProx_loc', 'Scaffold', 'Local', 'Centralized'],
                        help='Methods for the running experiment')
    parser.add_argument('--standardization', metavar='-std', type=str, default='static', choices = ['dynamic', 'static'],
                        help='Dynamic: the mean and std are updated during fed-miwae training. Static: the mean and std are evaluated previously to fed-miwae')
    parser.add_argument('--kind', metavar='-k', type=str, default='single', choices = ['single', 'multiple', 'both'],
                        help='Kind of imputation')
    parser.add_argument('--task', metavar='-ts', type=str, default='imputation', choices = ['imputation', 'prediction'],
                        help='Task to be performed with the pipeline')
    parser.add_argument('--Test_id', metavar='-tid', type=int, default=None,
                        help='Id of the Test dataset if any')
    parser.add_argument('--mask_num', metavar='-mkn', type=int, default=1,
                        help='Number of the simulated mask')
    parser.add_argument('--tags', metavar='-t', type=str, default='traumabase', help='Dataset tags', choices = ['traumabase','traumabase_imp','traumabase_pred'])
    parser.add_argument('--Rounds', metavar='-r', type=int, default=100,
                        help='Number of rounds for imputation')
    parser.add_argument('--Epochs', metavar='-e', type=int, default=None,
                        help='Number of epochs for imputation')
    parser.add_argument('--Updates', metavar='-up', type=int, default=None,
                        help='Number of updates')
    parser.add_argument('--data_folder', metavar='-d', type=str, default='data/',
                        help='Datasets folder')
    parser.add_argument('--root_data_folder', metavar='-rdf', type=str, default=None, choices=['fedbiomed','home'],
                        help='Root directory for data')
    parser.add_argument('--result_folder', metavar='-rf', type=str, default='results', 
                        help='Folder cotaining the results csv')
    parser.add_argument('--hidden', metavar='-h', type=int, default=128,
                        help='Number of hidden units')
    parser.add_argument('--latent', metavar='-d', type=int, default=3,
                        help='Latent dimension')
    parser.add_argument('--K', metavar='-k', type=int, default=50,
                        help='Number of IS during training')
    parser.add_argument('--L', metavar='-l', type=int, default=10000,
                        help='Number of samples in testing')
    parser.add_argument('--num_samples', metavar='-ns', type=int, default=30,
                        help='Number of multiple imputations (if kind == both or kind == multiple)')
    parser.add_argument('--stat_mul', metavar='-sm', type=str, default='mean_MSE', choices=['mean_MSE','MSE_of_mean'],
                        help='Statistics used for multiple imputation')
    parser.add_argument('--batch_size', metavar='-bs', type=int, default=48,
                        help='Batch size')
    parser.add_argument('--learning_rate', metavar='-lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--do_figures', metavar='-fig', default=True, action=argparse.BooleanOptionalAction,
                        help='Generate and save figures during local training')

    args = parser.parse_args()

    method = args.method
    std_type = args.standardization
    task = args.task
    idx_Test_data = args.Test_id
    if args.Test_id is not None:
        idx_Test_data = int(idx_Test_data)
    tags = args.tags
    data_folder = args.data_folder
    root_dir = args.root_data_folder
    kind = args.kind
    num_samples = args.num_samples

    mask_num = args.mask_num

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

    ############################################################################
    num_covariates = 12 if task=='imputation' else 13 #13 14
    ############################################################################

    #Number of partecipating clients
    N_cl = len(dataset_size)

    ###########################################################
    # Recover full dataset and test dataset for testing phase #
    ###########################################################

    if idx_Test_data is not None:
        idx_clients=[*range(1,N_cl+2)]
        idx_clients.remove(idx_Test_data)
    else:
        idx_clients=[*range(1,N_cl+1)]

    if task == 'imputation':
        if idx_Test_data is not None:
            Clients_data, Clients_missing, Clients_mask, data_test, data_test_missing, test_mask = databases(data_folder,task,idx_clients,mask_num,root_dir,idx_Test_data)
        else:
            Clients_data, Clients_missing, Clients_mask = databases(data_folder,task,idx_clients,mask_num,root_dir)
    elif task == 'prediction':
        Clients_data, Clients_label, data_test, test_label = databases(data_folder,task,idx_clients,mask_num,root_dir,idx_Test_data)
        Clients_missing = Clients_data.copy()
        data_test_missing = data_test.copy()
        Clients_mask = [np.isfinite(np.copy(cl)) for cl in Clients_data]
        test_mask = np.isfinite(np.copy(data_test))
    features = Clients_data[0].columns.values.tolist()
    #print(features)

    ###########################################################
    # Recover global mean and std in a federated manner       #
    ###########################################################

    fed_mean, fed_std = None, None

    if method not in ['FedProx_loc','Local', 'Centralized']:

        from fedbiomed.researcher.aggregators.fedstandard import FedStandard

        # NOTE: we need to perform only 1 round of 1 epoch to recover global mean and std
        ############################################################################
        model_args = {'n_features': data_size, 'n_cov': num_covariates}
        ############################################################################

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

    ###########################################################
    #First step: imputation of missing data                   #
    ###########################################################

    ###########################################################
    #Define the hyperparameters for miwae                     #
    ###########################################################

    h = args.hidden # number of hidden units in (same for all MLPs)
    d = args.latent # dimension of the latent space
    K = args.K # number of IS during training
    L = args.L # for testing phase

    batch_size = min(args.batch_size,min_n_samples)
    rounds = args.Rounds
    n_epochs=args.Epochs
    num_updates=args.Updates
    if ((n_epochs==None) and (num_updates==None)):
        print('Either the number of Epochs or the number of Updates has to be defined, both are None')
        sys.exit()
    #if num_updates == None:
    #    num_updates = int(np.ceil(min_n_samples/batch_size)*n_epochs)

    ###########################################################
    #Define the federated miwae model                         #
    ###########################################################

    if method not in ['Local','Centralized']:
        standardization = {'type':std_type}
        if method != 'FedProx_loc':
            standardization.update({'fed_mean':fed_mean.tolist(),'fed_std':fed_std.tolist()})

        model_args = {'n_features':data_size, 'n_cov': num_covariates, 'n_latent':d,'n_hidden':h,
                    'n_samples':K, 'n_samples_test':100, 'use_gpu': True, 'standardization':standardization}

        training_args = {
            'loader_args': {
                'batch_size': batch_size,
                'drop_last': True
            },
            'optimizer_args':
            {'lr': args.learning_rate}, 
            'log_interval' : 1,
          #  'num_updates': num_updates, 
            'dry_run': False
        }

        if n_epochs is not None:
            training_args.update(epochs = n_epochs)
        elif num_updates is not None:
            training_args.update(num_updates = num_updates)
            
        ###########################################################
        #Declare the experiment                                   #
        ###########################################################

        from fedbiomed.researcher.aggregators.fedavg import FedAverage
        from fedbiomed.researcher.aggregators.fedavg_fedstd import FedAverage_FedStd
        from fedbiomed.researcher.aggregators.scaffold import Scaffold

        if method == 'Scaffold':
            aggregator = Scaffold()  
        else:
            aggregator = FedAverage() if std_type=='static' else FedAverage_FedStd

        if 'fedprox_mu' in training_args:
            del training_args['fedprox_mu']

        exp = Experiment(tags=tags,
                        model_args=model_args,
                        training_plan_class=MIWAETrainingPlan,
                        training_args=training_args,
                        round_limit=rounds,
                        aggregator=aggregator,
                        node_selection_strategy=None)

        exp.run_once()

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

        if args.do_figures==True:
            Loss_cls = [[] for _ in range(N_cl)]
            Like_cls = [[] for _ in range(N_cl)]
            MSE_cls = [[] for _ in range(N_cl)]

        # Recall all hyperparameters
        n_epochs_local = n_epochs*rounds

        bs = args.batch_size
        lr = args.learning_rate

        h = args.hidden
        d = args.latent
        K = args.K 

        Encoders_loc = []
        Decoders_loc = []
        Iota_loc = []

        for cls in range(N_cl):
            # Data
            n = Clients_data[cls].shape[0] # number of observations
            p = Clients_data[cls].shape[1]-num_covariates # number of features

            xhat_local_std, xfull_local_std, local_mean, local_std = recover_data(Clients_missing[cls], Clients_data[cls], num_covariates-1)

            if task == 'imputation':            
                mask_cls_test = np.copy(Clients_mask[cls])

            xhat_cls = np.copy(xhat_local_std)
            mask_cls = np.isfinite(xhat_cls)[:,num_covariates:]
            xhat_0_cls = np.copy(xhat_local_std)[:,num_covariates:]
            x_cat_cls = np.copy(xhat_local_std)[:,:num_covariates]
            xfull_cls = np.copy(xfull_local_std)

            # Model
            encoder_cls, decoder_cls, iota_cls, optimizer_cls = encoder_decoder_iota_opt(p,num_covariates,h,d,lr)

            for ep in range(1,n_epochs_local):
                perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
                batches_data = np.array_split(xhat_0_cls[perm,], n/bs)
                batches_mask = np.array_split(mask_cls[perm,], n/bs)
                batches_x_cat = np.array_split(x_cat_cls[perm,], n/bs)
                for it in range(len(batches_data)):
                    optimizer_cls.zero_grad()
                    encoder_cls.zero_grad()
                    decoder_cls.zero_grad()
                    b_data = torch.from_numpy(batches_data[it]).float()
                    b_mask = torch.from_numpy(batches_mask[it]).float()
                    b_cat = torch.from_numpy(batches_x_cat[it]).float()
                    loss = miwae_loss(encoder = encoder_cls,decoder = decoder_cls, iota=iota_cls,
                                    data = b_data, xcat=b_cat, mask = b_mask, d = d, p = p, K = K)
                    loss.backward()
                    optimizer_cls.step()
                if ((args.do_figures==True) and (task == 'imputation')):
                    likelihood = (-np.log(K)-miwae_loss(encoder = encoder_cls,decoder = decoder_cls,iota=iota_cls, 
                            data = torch.from_numpy(xhat_0_cls).float(), xcat=torch.from_numpy(x_cat_cls).float(), mask = torch.from_numpy(mask_cls).float(), 
                            d = d, p = p, K = K).cpu().data.numpy())
                    Loss_cls[cls].append(loss.item())
                    Like_cls[cls].append(likelihood)
                    mse_train = testing_func(xhat_local_std[:,num_covariates:], xfull_local_std[:,num_covariates:], x_cat_cls, 
                                            mask_cls_test[:,num_covariates:], encoder_cls, decoder_cls, iota_cls, d, 100)
                    MSE_cls[cls].append(mse_train)
                if ep % rounds == 1:
                    likelihood = (-np.log(K)-miwae_loss(encoder = encoder_cls,decoder = decoder_cls,iota=iota_cls, 
                            data = torch.from_numpy(xhat_0_cls).float(), xcat=torch.from_numpy(x_cat_cls).float(), mask = torch.from_numpy(mask_cls).float(), 
                            d = d, p = p, K = K).cpu().data.numpy())
                    print('Epoch %g' %ep)
                    print('MIWAE likelihood bound:  %g' %likelihood) # Gradient step      
                    print('Loss: {:.6f}'.format(loss.item()))
                    if ((args.do_figures==True) and (task == 'imputation')):
                        print('MSE:  %g' %mse_train)

            Encoders_loc.append(encoder_cls)
            Decoders_loc.append(decoder_cls)
            Iota_loc.append(iota_cls)

    elif method == 'Centralized':
        # Recall all hyperparameters
        n_epochs_centralized = n_epochs*rounds*N_cl

        bs = args.batch_size
        lr = args.learning_rate

        h = args.hidden
        d = args.latent
        K = args.K 
        # Centralized training
        if args.do_figures==True:
            Loss_tot = []
            Like_tot = []
            MSE_tot = []

        xmiss_tot = np.concatenate(Clients_missing,axis=0)
        mask_tot_test = np.concatenate(Clients_mask,axis=0)
        x_cat_tot = np.copy(xmiss_tot)[:,:num_covariates]
        ###########################################################
        ##Fill nan values with mean
        #xmiss_tot_filled = np.copy(xmiss_tot)
        #col_mean = np.nanmean(xmiss_tot_filled,0)
        ##Find indices that you need to replace
        #inds = np.where(np.isnan(xmiss_tot_filled))
        ##Place column means in the indices. Align the arrays using take
        #xmiss_tot_filled[inds] = np.take(col_mean, inds[1])
        ###########################################################

        n = xmiss_tot.shape[0] # number of observations
        p = xmiss_tot.shape[1]-num_covariates # number of features

        ###########################################################
        mean_tot_missing = np.nanmean(xmiss_tot[:,num_covariates-1:],0)
        std_tot_missing = np.nanstd(xmiss_tot[:,num_covariates-1:],0)
        ###########################################################
        xmiss_tot = np.concatenate((xmiss_tot[:,:num_covariates-1], (xmiss_tot[:,num_covariates-1:] - mean_tot_missing)/std_tot_missing), axis=1)
        mask_tot = np.isfinite(xmiss_tot)[:,num_covariates:] # binary mask that indicates which values are missing
        xhat_0_tot = np.copy(xmiss_tot)
        xhat_0_tot[np.isnan(xmiss_tot)] = 0
        xhat_0_tot = xhat_0_tot[:,num_covariates:]
        xhat_tot = np.copy(xhat_0_tot) # This will be out imputed data matrix

        xfull_tot = np.concatenate(Clients_data,axis=0)
        xfull_tot = np.concatenate((xfull_tot[:,:num_covariates-1], (xfull_tot[:,num_covariates-1:] - mean_tot_missing)/std_tot_missing), axis=1)

        # Model
        encoder_cen, decoder_cen, iota_cen, optimizer_cen = encoder_decoder_iota_opt(p,num_covariates,h,d,lr)

        # Training loop

        for ep in range(1,n_epochs_centralized):
            perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
            batches_data = np.array_split(xhat_0_tot[perm,], n/bs)
            batches_mask = np.array_split(mask_tot[perm,], n/bs)
            batches_x_cat = np.array_split(x_cat_tot[perm,], n/bs)
            for it in range(len(batches_data)):
                optimizer_cen.zero_grad()
                encoder_cen.zero_grad()
                decoder_cen.zero_grad()
                b_data = torch.from_numpy(batches_data[it]).float()
                b_mask = torch.from_numpy(batches_mask[it]).float()
                b_cat = torch.from_numpy(batches_x_cat[it]).float()
                loss = miwae_loss(encoder = encoder_cen,decoder = decoder_cen, iota=iota_cen, data = b_data, xcat=b_cat,mask = b_mask, d = d, p = p, K = K)
                loss.backward()
                optimizer_cen.step()
            if ((args.do_figures==True) and (task == 'imputation')):
                likelihood = (-np.log(K)-miwae_loss(encoder = encoder_cen,decoder = decoder_cen,iota=iota_cen, 
                            data = torch.from_numpy(xhat_0_tot).float(), xcat=torch.from_numpy(x_cat_tot).float(),
                            mask = torch.from_numpy(mask_tot).float(), d = d, p = p, K = K).cpu().data.numpy())
                Loss_tot.append(loss.item())
                Like_tot.append(likelihood)
                mse_train = testing_func(xhat_0_tot, xfull_tot[:,num_covariates:], x_cat_tot, mask_tot_test[:,num_covariates:], 
                                        encoder_cen, decoder_cen, iota_cen, d, 100)#,kind,num_samples)
                MSE_tot.append(mse_train)
            if ep % rounds == 1:
                likelihood = (-np.log(K)-miwae_loss(encoder = encoder_cen,decoder = decoder_cen,iota=iota_cen, 
                            data = torch.from_numpy(xhat_0_tot).float(), xcat=torch.from_numpy(x_cat_tot).float(),
                            mask = torch.from_numpy(mask_tot).float(), d = d, p = p, K = K).cpu().data.numpy())
                print('Epoch %g' %ep)
                print('MIWAE likelihood bound:  %g' %likelihood) # Gradient step      
                print('Loss: {:.6f}'.format(loss.item()))
                if ((args.do_figures==True) and (task == 'imputation')):
                    print('MSE:  %g' %mse_train)

    ###########################################################
    #Recover model and params                                 #
    ###########################################################

    result_folder = args.result_folder

    if method not in ['Local','Centralized']:
        # extract federated model into PyTorch framework
        model = exp.training_plan().model()
        model.load_state_dict(exp.aggregated_params()[rounds - 1]['params'])
        encoder = model.encoder
        decoder = model.decoder
        iota = model.iota
        std_training = 'Loc' if method == 'FedProx_loc' else 'Fed'
        if std_type=='dynamic':
            fed_mean = model.mean
            fed_std = model.std
    elif method == 'Centralized':
        encoder,decoder,iota = encoder_cen,decoder_cen,iota_cen
        fed_mean, fed_std = mean_tot_missing, std_tot_missing

    ###########################################################
    #Testing phase (imputation)                               #
    ###########################################################

    if task == 'imputation':
        stat_mul = args.stat_mul if kind != 'single' else None
        if ((method == 'Local') and (args.do_figures==True)):
            generate_save_plots(result_folder,Loss_cls,Like_cls,MSE_cls,n_epochs_local,method,idx_clients)
        elif ((method == 'Centralized') and (args.do_figures==True)):
            generate_save_plots(result_folder,Loss_tot,Like_tot,MSE_tot,n_epochs_centralized,method,idx_clients)                   
        # Testing on data used during training
        for cls in range(N_cl):
            ###########################################################
            if ((fed_mean is None) and (fed_std is None)):
                xhat_local_std, xfull_local_std, loc_mean, loc_std = recover_data(Clients_missing[cls], Clients_data[cls], num_covariates)
            else:
                xhat_global_std, xfull_global_std, xhat_local_std, xfull_local_std, loc_mean, loc_std =\
                    recover_data(Clients_missing[cls], Clients_data[cls], num_covariates, fed_mean, fed_std)
            ###########################################################
            mask = np.copy(Clients_mask[cls])[:,num_covariates:]
            x_cat = xhat_local_std[:,:num_covariates]
            if method not in ['Local','Centralized']:
                MSE = testing_func_mul(features=features, data_missing=xhat_local_std[:,num_covariates:], 
                                       data_full=xfull_local_std[:,num_covariates:], x_cat=x_cat, mask=mask, 
                                       encoder=encoder, decoder=decoder, iota=iota, d=d, L=L,
                                       idx_cl=idx_clients[cls],result_folder=result_folder,
                                       method=method,kind=kind,num_samples=num_samples,
                                       mean=loc_mean,std=loc_std,do_figures=args.do_figures, stat_mul=stat_mul)
                save_results_imputation(result_folder, mask_num,idx_clients,idx_clients[cls],method,
                    N_cl,[len(Clients_missing[i]) for i in range(N_cl)],rounds,n_epochs,
                    std_training,'local',MSE)
                if method != 'FedProx_loc':
                    MSE = testing_func_mul(features=features, data_missing=xhat_global_std[:,num_covariates:], 
                                       data_full=xfull_global_std[:,num_covariates:], x_cat=x_cat, mask=mask, 
                                       encoder=encoder, decoder=decoder, iota=iota, d=d, L=L,
                                       idx_cl=idx_clients[cls],result_folder=result_folder,
                                       method=method,kind=kind,num_samples=num_samples,
                                       mean=fed_mean,std=fed_std,do_figures=args.do_figures, stat_mul=stat_mul)
                    save_results_imputation(result_folder, mask_num,idx_clients,idx_clients[cls],method,
                        N_cl,[len(Clients_missing[i]) for i in range(N_cl)],rounds,n_epochs,
                        std_training,'global',MSE)
            elif method == 'Centralized':
                # centralized 
                MSE = testing_func_mul(features=features, data_missing=xhat_local_std[:,num_covariates:], 
                                       data_full=xfull_local_std[:,num_covariates:], x_cat=x_cat, mask=mask, 
                                       encoder=encoder_cen, decoder=decoder_cen, iota=iota_cen, d=d, L=L,
                                       idx_cl=idx_clients[cls],result_folder=result_folder,
                                       method='Centralized',kind=kind,num_samples=num_samples,
                                       mean=loc_mean,std=loc_std,do_figures=args.do_figures, stat_mul=stat_mul)
                save_results_imputation(result_folder, mask_num,sum(idx_clients),idx_clients[cls],'Centralized',
                    1,[len(xmiss_tot)],1,n_epochs_centralized,
                    'Loc','local',MSE)
                MSE = testing_func_mul(features=features, data_missing=xhat_global_std[:,num_covariates:], 
                                       data_full=xfull_global_std[:,num_covariates:], x_cat=x_cat, mask=mask, 
                                       encoder=encoder_cen, decoder=decoder_cen, iota=iota_cen, d=d, L=L,
                                       idx_cl=idx_clients[cls],result_folder=result_folder,
                                       method='Centralized',kind=kind,num_samples=num_samples,
                                       mean=fed_mean,std=fed_std,do_figures=args.do_figures, stat_mul=stat_mul)
                save_results_imputation(result_folder, mask_num,sum(idx_clients),idx_clients[cls],'Centralized',
                    1,[len(xmiss_tot)],1,n_epochs_centralized,
                    'Loc','global',MSE)
            elif method == 'Local':
                # local
                MSE = testing_func_mul(features=features, data_missing=xhat_local_std[:,num_covariates:], 
                                       data_full=xfull_local_std[:,num_covariates:], x_cat=x_cat, mask=mask, 
                                       encoder=Encoders_loc[cls], decoder=Decoders_loc[cls], iota=Iota_loc[cls], d=d, L=L,
                                       idx_cl=idx_clients[cls],result_folder=result_folder,
                                       method='Local_cl'+str(idx_clients[cls]),kind=kind,num_samples=num_samples,
                                       mean=loc_mean,std=loc_std,do_figures=args.do_figures, stat_mul=stat_mul)
                save_results_imputation(result_folder, mask_num,idx_clients[cls],idx_clients[cls],'Local_cl'+str(idx_clients[cls]),
                    1,[len(Clients_missing[cls])],1,n_epochs_local,
                    'Loc','local',MSE)

        # Testing on external dataset
        ###########################################################
        if idx_Test_data is not None:
            if ((fed_mean is None) and (fed_std is None)):
                xhat_local_std, xfull_local_std, loc_mean, loc_std = recover_data(data_test_missing, data_test, num_covariates)
            else:
                xhat_global_std, xfull_global_std, xhat_local_std, xfull_local_std, loc_mean, loc_std =\
                        recover_data(data_test_missing, data_test, num_covariates, fed_mean, fed_std)
            ###########################################################
            mask = np.copy(test_mask)[:,num_covariates:]
            x_cat = xhat_local_std[:,:num_covariates]
            if method not in ['Local','Centralized']:
                MSE = testing_func_mul(features=features, data_missing=xhat_local_std[:,num_covariates:], 
                                        data_full=xfull_local_std[:,num_covariates:], x_cat=x_cat, mask=mask, 
                                        encoder=encoder, decoder=decoder, iota=iota, d=d, L=L,
                                        idx_cl=idx_Test_data,result_folder=result_folder,
                                        method=method,kind=kind,num_samples=num_samples,
                                        mean=loc_mean,std=loc_std,do_figures=args.do_figures, stat_mul=stat_mul)
                save_results_imputation(result_folder, mask_num,idx_clients,idx_Test_data,method,
                    N_cl,[len(Clients_missing[i]) for i in range(N_cl)],rounds,n_epochs,
                    std_training,'local',MSE)
                if method != 'FedProx_loc':
                    MSE = testing_func_mul(features=features, data_missing=xhat_global_std[:,num_covariates:], 
                                        data_full=xfull_global_std[:,num_covariates:], x_cat=x_cat, mask=mask, 
                                        encoder=encoder, decoder=decoder, iota=iota, d=d, L=L,
                                        idx_cl=idx_Test_data,result_folder=result_folder,
                                        method=method,kind=kind,num_samples=num_samples,
                                        mean=fed_mean,std=fed_std,do_figures=args.do_figures, stat_mul=stat_mul)
                    save_results_imputation(result_folder, mask_num,idx_clients,idx_Test_data,method,
                        N_cl,[len(Clients_missing[i]) for i in range(N_cl)],rounds,n_epochs,
                        std_training,'global',MSE)
            elif method == 'Centralized':
                # centralized 
                MSE = testing_func_mul(features=features, data_missing=xhat_local_std[:,num_covariates:], 
                                        data_full=xfull_local_std[:,num_covariates:], x_cat=x_cat, mask=mask, 
                                        encoder=encoder_cen, decoder=decoder_cen, iota=iota_cen, d=d, L=L,
                                        idx_cl=idx_Test_data,result_folder=result_folder,
                                        method='Centralized',kind=kind,num_samples=num_samples,
                                        mean=loc_mean,std=loc_std,do_figures=args.do_figures, stat_mul=stat_mul)
                save_results_imputation(result_folder, mask_num,sum(idx_clients),idx_Test_data,'Centralized',
                    1,[len(xmiss_tot)],1,n_epochs_centralized,
                    'Loc','local',MSE)
                MSE = testing_func_mul(features=features, data_missing=xhat_global_std[:,num_covariates:], 
                                        data_full=xfull_global_std[:,num_covariates:], x_cat=x_cat, mask=mask, 
                                        encoder=encoder_cen, decoder=decoder_cen, iota=iota_cen, d=d, L=L,
                                        idx_cl=idx_Test_data,result_folder=result_folder,
                                        method='Centralized',kind=kind,num_samples=num_samples,
                                        mean=fed_mean,std=fed_std,do_figures=args.do_figures, stat_mul=stat_mul)
                save_results_imputation(result_folder, mask_num,sum(idx_clients),idx_Test_data,'Centralized',
                    1,[len(xmiss_tot)],1,n_epochs_centralized,
                    'Loc','global',MSE)
            elif method == 'Local':
                # local
                for cls in range(N_cl):
                    MSE = testing_func_mul(features=features, data_missing=xhat_local_std[:,num_covariates:], 
                                        data_full=xfull_local_std[:,num_covariates:], x_cat=x_cat, mask=mask, 
                                        encoder=Encoders_loc[cls], decoder=Decoders_loc[cls], iota=Iota_loc[cls], d=d, L=L,
                                        idx_cl=idx_Test_data,result_folder=result_folder,
                                        method='Local_cl'+str(idx_clients[cls]),kind=kind,num_samples=num_samples,
                                        mean=loc_mean,std=loc_std,do_figures=args.do_figures, stat_mul=stat_mul)
                    save_results_imputation(result_folder, mask_num,idx_clients[cls],idx_Test_data,'Local_cl'+str(idx_clients[cls]),
                        1,[len(Clients_missing[cls])],1,n_epochs_local,
                        'Loc','local',MSE)

    ###########################################################
    #Second step: create imputed dataset                      #
    ###########################################################

    elif task == 'prediction':
        #np.savez_compressed('/'+result_folder+'/clients_imputed/'+method+'_mean_std', mean=fed_mean, std=fed_std)
        #num_covariates -= 1
        for cls in range(N_cl):
            filename = 'Client_imputed_' +str(idx_clients[cls])
            # Fill clients missing data
            if method == 'Local':
                create_save_data_prediction(Encoders_loc[cls], Decoders_loc[cls], Iota_loc[cls], Clients_data[cls], num_covariates, \
                    result_folder, filename, d, L, standard = True, kind=kind, num_samples=num_samples)
            else:
                create_save_data_prediction(encoder, decoder, iota, Clients_data[cls], num_covariates, \
                    result_folder, filename, d, L, standard = True, mean = fed_mean, std = fed_std, kind=kind, num_samples=num_samples)
        #Impute test missing data
        filename = 'Client_imputed_' +str(idx_Test_data)
        if method == 'Local':
            create_save_data_prediction(Encoders_loc[cls], Decoders_loc[cls], Iota_loc[cls], data_test, num_covariates, \
                result_folder, filename, d, L, standard = True, kind=kind, num_samples=num_samples)
        else:
            create_save_data_prediction(encoder, decoder, iota, data_test, num_covariates, \
                result_folder, filename, d, L, standard = True, mean = fed_mean, std = fed_std, kind=kind, num_samples=num_samples)
                            


