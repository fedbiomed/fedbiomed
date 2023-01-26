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
import torch.nn as nn
import argparse
import sys

from func_miwae_adni import miwae_loss, recover_data, testing_func, save_results, databases
from class_miwae_adni import FedMeanStdTrainingPlan, MIWAETrainingPlan

from fedbiomed.researcher.experiment import Experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline fed standardization and imputation ADNI')
    parser.add_argument('--scenario', metavar='-s', type=str, default='site_1', choices = ['site_1', 'site_2', 'notiid'],
                        help='Scenario for data splitting')
    parser.add_argument('--method', metavar='-m', type=str, default='FedAvg', choices = ['FedAvg', 'FedProx', 'FedProx_loc', 'Scaffold', 'Local'],
                        help='Methods for the running experiment')
    parser.add_argument('--Test_id', metavar='-tid', type=int, default=4,
                        help='Id of the Test dataset (between 1 and 4)')
    parser.add_argument('--tags', metavar='-t', type=str, default='adni_1', choices = ['adni_1', 'adni_2', 'adni_notiid'],
                        help='Dataset tags')
    parser.add_argument('--Rounds', metavar='-r', type=int, default=200,
                        help='Number of rounds')
    parser.add_argument('--Epochs', metavar='-e', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--data_folder', metavar='-d', type=str, default='data/',
                        help='Datasets folder')
    parser.add_argument('--root_data_folder', metavar='-rdf', type=str, default=None, choices=['fedbiomed','home'],
                        help='Root directory for data')
    parser.add_argument('--result_folder', metavar='-rf', type=str, default='results', 
                        help='Folder cotaining the results csv')
    parser.add_argument('--hidden', metavar='-h', type=int, default=256,
                        help='Number of epochs')
    parser.add_argument('--latent', metavar='-d', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--K', metavar='-k', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--L', metavar='-l', type=int, default=10000,
                        help='Number of epochs')
    parser.add_argument('--batch_size', metavar='-bs', type=int, default=32,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', metavar='-lr', type=float, default=1e-3,
                        help='Learning rate')

    args = parser.parse_args()

    Split_type = args.scenario
    method = args.method
    idx_Test_data = int(args.Test_id)
    tags = args.tags
    data_folder = args.data_folder
    root_dir = args.root_data_folder

    if ((Split_type=='site_1') and (tags!='adni_1')):
        print('Split tipe and tags do not match:', Split_type,tags)
        sys.exit()
    elif ((Split_type=='site_2') and (tags!='adni_2')):
        print('Split tipe and tags do not match:', Split_type,tags)
        sys.exit()
    elif ((Split_type=='notiid') and (tags!='adni_notiid')):
        print('Split tipe and tags do not match:', Split_type,tags)
        sys.exit()

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

    #Number of partecipating clients
    N_cl = len(dataset_size)

    ###########################################################
    # Recover full dataset and test dataset for testing phase #
    ###########################################################

    idx_clients=[*range(1,N_cl+2)]
    idx_clients.remove(idx_Test_data)

    Clients_data, Clients_missing, data_test, data_test_missing, Perc_missing, Perc_missing_test = databases(data_folder,Split_type,idx_clients,idx_Test_data,N_cl,root_dir)

    ###########################################################
    # Recover global mean and std in a federated manner       #
    ###########################################################

    fed_mean, fed_std = None, None

    if method not in ['FedProx_loc','Local']:

        from fedbiomed.researcher.aggregators.fedstandard import FedStandard

        # NOTE: we need to perform only 1 round of 1 epoch to recover global mean and std
        model_args = {'n_features':data_size}

        training_args = {
            'batch_size': 48, 
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
        fed_mean = fed_mean_std.aggregated_params()[0]['params']['fed_mean']
        fed_std = fed_mean_std.aggregated_params()[0]['params']['fed_std']

    ###########################################################
    #Define the hyperparameters for miwae                     #
    ###########################################################

    h = args.hidden # number of hidden units in (same for all MLPs)
    d = args.latent # dimension of the latent space
    K = args.K # number of IS during training
    L = args.L # for testing phase

    n_epochs=args.Epochs
    batch_size = args.batch_size
    num_updates = int(np.ceil(min_n_samples/batch_size)*n_epochs)
    rounds = args.Rounds

    ###########################################################
    #Define the federated miwae model                         #
    ###########################################################

    if method != 'Local':

        standardization = {} if method == 'FedProx_loc' else {'fed_mean':fed_mean.tolist(),'fed_std':fed_std.tolist()}

        model_args = {'n_features':data_size, 'n_latent':d,'n_hidden':h,'n_samples':K, 'use_gpu': True,
                    'standardization':standardization}

        training_args = {
            'batch_size': batch_size, 
            'optimizer_args':
            {'lr': args.learning_rate}, 
            'log_interval' : 1,
            'num_updates': num_updates, 
            'dry_run': False
        }
            
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
        # Recall all hyperparameters
        n_epochs_local = n_epochs*rounds
        n_epochs_centralized = n_epochs*rounds*len(Clients_missing)

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
            p = Clients_data[cls].shape[1] # number of features

            xmiss, mask, xhat_local_std, xfull_local_std =\
                 recover_data(Clients_missing[cls], Clients_data[cls])

            xhat_cls = np.copy(xhat_local_std)
            xhat_0_cls = np.copy(xhat_local_std)
            xfull_cls = np.copy(xfull_local_std)
            mask_cls = np.copy(mask)

            encoder_cls = nn.Sequential(
                torch.nn.Linear(p, h),
                #torch.nn.ReLU(),
                #torch.nn.Linear(h, h),
                torch.nn.ReLU(),
                torch.nn.Linear(h, 3*d),  
            )

            decoder_cls = nn.Sequential(
                torch.nn.Linear(d, h),
                #torch.nn.ReLU(),
                #torch.nn.Linear(h, h),
                torch.nn.ReLU(),
                torch.nn.Linear(h, 3*p),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
            )

            iota_cls = nn.Parameter(torch.zeros(1,p),requires_grad=True)

            optimizer_cls = torch.optim.Adam(list(encoder_cls.parameters()) + list(decoder_cls.parameters()) + [iota_cls],lr=1e-3)

            def weights_init(layer):
                if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
                    
            encoder_cls.apply(weights_init)
            decoder_cls.apply(weights_init)

            for ep in range(1,n_epochs_local):
                perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
                batches_data = np.array_split(xhat_0_cls[perm,], n/bs)
                batches_mask = np.array_split(mask_cls[perm,], n/bs)
                for it in range(len(batches_data)):
                    optimizer_cls.zero_grad()
                    encoder_cls.zero_grad()
                    decoder_cls.zero_grad()
                    b_data = torch.from_numpy(batches_data[it]).float()
                    b_mask = torch.from_numpy(batches_mask[it]).float()
                    loss = miwae_loss(encoder = encoder_cls,decoder = decoder_cls, iota=iota_cls,
                                    data = b_data,mask = b_mask, d = d, p = p, K = K)
                    loss.backward()
                    optimizer_cls.step()
                if ep % rounds == 1:
                    print('Epoch %g' %ep)
                    print('MIWAE likelihood bound  %g' %(-np.log(K)-miwae_loss(encoder = encoder_cls,decoder = decoder_cls,iota=iota_cls, data = torch.from_numpy(xhat_0_cls).float(),mask = torch.from_numpy(mask_cls).float(), d = d, p = p, K = K).cpu().data.numpy())) # Gradient step      
                    print('Loss: {:.6f}'.format(loss.item()))

            Encoders_loc.append(encoder_cls)
            Decoders_loc.append(decoder_cls)
            Iota_loc.append(iota_cls)

        # Centralized training
        xmiss_tot = np.concatenate(Clients_missing,axis=0)

        n = xmiss_tot.shape[0] # number of observations
        p = xmiss_tot.shape[1] # number of features

        mean_tot_missing = np.nanmean(xmiss_tot,0)
        std_tot_missing = np.nanstd(xmiss_tot,0)
        xmiss_tot = (xmiss_tot - mean_tot_missing)/std_tot_missing
        mask_tot = np.isfinite(xmiss_tot) # binary mask that indicates which values are missing
        xhat_0_tot = np.copy(xmiss_tot)
        xhat_0_tot[np.isnan(xmiss_tot)] = 0
        xhat_tot = np.copy(xhat_0_tot) # This will be out imputed data matrix

        xfull_tot = np.concatenate(Clients_data,axis=0)
        xfull_tot = (xfull_tot - mean_tot_missing)/std_tot_missing

        # Model

        encoder_cen = nn.Sequential(
            torch.nn.Linear(p, h),
            #torch.nn.ReLU(),
            #torch.nn.Linear(h, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, 3*d),  # the encoder will output both the mean and the diagonal covariance
        )

        decoder_cen = nn.Sequential(
            torch.nn.Linear(d, h),
            #torch.nn.ReLU(),
            #torch.nn.Linear(h, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, 3*p),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
        )

        iota_cen = nn.Parameter(torch.zeros(1,p),requires_grad=True)

        optimizer_cen = torch.optim.Adam(list(encoder_cen.parameters()) + list(decoder_cen.parameters()) + [iota_cen],lr=1e-3)

        def weights_init(layer):
            if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
                
        encoder_cen.apply(weights_init)
        decoder_cen.apply(weights_init)

        # Training loop

        for ep in range(1,n_epochs_centralized):
            perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
            batches_data = np.array_split(xhat_0_tot[perm,], n/bs)
            batches_mask = np.array_split(mask_tot[perm,], n/bs)
            for it in range(len(batches_data)):
                optimizer_cen.zero_grad()
                encoder_cen.zero_grad()
                decoder_cen.zero_grad()
                b_data = torch.from_numpy(batches_data[it]).float()
                b_mask = torch.from_numpy(batches_mask[it]).float()
                loss = miwae_loss(encoder = encoder_cen,decoder = decoder_cen, iota=iota_cls, data = b_data,mask = b_mask, d = d, p = p, K = K)
                loss.backward()
                optimizer_cen.step()
            if ep % rounds == 1:
                print('Epoch %g' %ep)
                print('MIWAE likelihood bound  %g' %(-np.log(K)-miwae_loss(encoder = encoder_cen,decoder = decoder_cen,iota=iota_cls, data = torch.from_numpy(xhat_0_tot).float(),mask = torch.from_numpy(mask_tot).float(), d = d, p = p, K = K).cpu().data.numpy())) # Gradient step      
                print('Loss: {:.6f}'.format(loss.item()))

    ###########################################################
    #Testing phase                                            #
    ###########################################################
    result_folder = args.result_folder

    if method != 'Local':
        # extract federated model into PyTorch framework
        model = exp.training_plan().model()
        model.load_state_dict(exp.aggregated_params()[rounds - 1]['params'])
        encoder = model.encoder
        decoder = model.decoder
        iota = model.iota
        std_training = 'Loc' if method == 'FedProx_loc' else 'Fed'
    else:
        fed_mean, fed_std = mean_tot_missing, std_tot_missing

    # Testing on data used during training
    for cls in range(N_cl):
        if ((fed_mean is None) and (fed_std is None)):
            xmiss, mask, xhat_local_std, xfull_local_std =\
                recover_data(Clients_missing[cls], Clients_data[cls])
        else:
            xmiss, mask, xhat_global_std, xfull_global_std, xhat_local_std, xfull_local_std =\
                recover_data(Clients_missing[cls], Clients_data[cls], fed_mean, fed_std)
        if method != 'Local':
            MSE = testing_func(xhat_local_std, xfull_local_std, mask, encoder, decoder, iota, d, L)
            save_results(result_folder,Split_type,idx_clients,idx_clients[cls],
                Perc_missing,Perc_missing[cls],method,
                N_cl,[len(Clients_missing[i]) for i in range(N_cl)],rounds,n_epochs,
                std_training,'local',MSE)
            if method != 'FedProx_loc':
                MSE = testing_func(xhat_global_std, xfull_global_std, mask, encoder, decoder, iota, d, L)
                save_results(result_folder,Split_type,idx_clients,idx_clients[cls],
                    Perc_missing,Perc_missing[cls],method,
                    N_cl,[len(Clients_missing[i]) for i in range(N_cl)],rounds,n_epochs,
                    std_training,'global',MSE)
        elif method == 'Local':
            # centralized 
            mean_tot_missing
            MSE = testing_func(xhat_local_std, xfull_local_std, mask, encoder_cen, decoder_cen, iota_cen, d, L)
            save_results(result_folder,Split_type,sum(idx_clients),idx_clients[cls],
                Perc_missing,Perc_missing[cls],'Centralized',
                1,[len(xmiss_tot)],1,n_epochs_centralized,
                'Loc','local',MSE)
            MSE = testing_func(xhat_global_std, xfull_global_std, mask, encoder_cen, decoder_cen, iota_cen, d, L)
            save_results(result_folder,Split_type,sum(idx_clients),idx_clients[cls],
                Perc_missing,Perc_missing[cls],'Centralized',
                1,[len(xmiss_tot)],1,n_epochs_centralized,
                'Loc','global',MSE)
            # local
            MSE = testing_func(xhat_local_std, xfull_local_std, mask, Encoders_loc[cls], Decoders_loc[cls], Iota_loc[cls], d, L)
            save_results(result_folder,Split_type,idx_clients[cls],idx_clients[cls],
                Perc_missing[cls],Perc_missing[cls],'Local_cl'+str(idx_clients[cls]),
                1,[len(Clients_missing[cls])],1,n_epochs_local,
                'Loc','local',MSE)

    # Testing on external dataset
    if ((fed_mean is None) and (fed_std is None)):
        xmiss, mask, xhat_local_std, xfull_local_std =\
                recover_data(data_test_missing, data_test)
    else:
        xmiss, mask, xhat_global_std, xfull_global_std, xhat_local_std, xfull_local_std =\
                recover_data(data_test_missing, data_test, fed_mean, fed_std)
    if method != 'Local':
        MSE = testing_func(xhat_local_std, xfull_local_std, mask, encoder, decoder, iota, d, L)
        save_results(result_folder,Split_type,idx_clients,idx_Test_data,
            Perc_missing,Perc_missing_test,method,
            N_cl,[len(Clients_missing[i]) for i in range(N_cl)],rounds,n_epochs,
            std_training,'local',MSE)
        if method != 'FedProx_loc':
            MSE = testing_func(xhat_global_std, xfull_global_std, mask, encoder, decoder, iota, d, L)
            save_results(result_folder,Split_type,idx_clients,idx_Test_data,
                Perc_missing,Perc_missing_test,method,
                N_cl,[len(Clients_missing[i]) for i in range(N_cl)],rounds,n_epochs,
                std_training,'global',MSE)
    elif method == 'Local':
        # centralized 
        MSE = testing_func(xhat_local_std, xfull_local_std, mask, encoder_cen, decoder_cen, iota_cen, d, L)
        save_results(result_folder,Split_type,sum(idx_clients),idx_Test_data,
            Perc_missing,Perc_missing_test,'Centralized',
            1,[len(xmiss_tot)],1,n_epochs_centralized,
            'Loc','local',MSE)
        MSE = testing_func(xhat_global_std, xfull_global_std, mask, encoder_cen, decoder_cen, iota_cen, d, L)
        save_results(result_folder,Split_type,sum(idx_clients),idx_Test_data,
            Perc_missing,Perc_missing_test,'Centralized',
            1,[len(xmiss_tot)],1,n_epochs_centralized,
            'Loc','global',MSE)
        # local
        for cls in range(N_cl):
            MSE = testing_func(xhat_local_std, xfull_local_std, mask, Encoders_loc[cls], Decoders_loc[cls], Iota_loc[cls], d, L)
            save_results(result_folder,Split_type,idx_clients[cls],idx_Test_data,
                Perc_missing[cls],Perc_missing_test,'Local_cl'+str(idx_clients[cls]),
                1,[len(Clients_missing[cls])],1,n_epochs_local,
                'Loc','local',MSE)


