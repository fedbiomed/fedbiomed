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

from func_miwae_adni import miwae_loss, recover_data, testing_func, save_results, databases, generate_save_plots, encoder_decoder_iota_opt
from class_miwae_adni import FedMeanStdTrainingPlan, MIWAETrainingPlan

from fedbiomed.researcher.experiment import Experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline fed standardization and imputation ADNI')
    parser.add_argument('--scenario', metavar='-s', type=str, default='site_1', choices = ['site_1', 'site_2', 'notiid'],
                        help='Scenario for data splitting')
    parser.add_argument('--method', metavar='-m', type=str, default='FedAvg', choices = ['FedAvg', 'FedProx', 'FedProx_loc', 'Scaffold', 'Local'],
                        help='Methods for the running experiment')
    parser.add_argument('--standardization', metavar='-std', type=str, default='dynamic', choices = ['dynamic', 'static', 'dyn_iota'],
                        help='Dynamic: the mean and std are updated during fed-miwae training. Static: the mean and std are evaluated previously to fed-miwae')
    parser.add_argument('--kind', metavar='-k', type=str, default='single', choices = ['single', 'multiple', 'both'],
                        help='Kind of imputation')
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
                        help='Number of hidden units')
    parser.add_argument('--latent', metavar='-d', type=int, default=20,
                        help='Latent dimension')
    parser.add_argument('--K', metavar='-k', type=int, default=50,
                        help='Number of IS during training')
    parser.add_argument('--L', metavar='-l', type=int, default=10000,
                        help='Number of IS during testing')
    parser.add_argument('--num_samples', metavar='-ns', type=int, default=30,
                        help='Number of multiple imputations (if kind == both or kind == multiple)')
    parser.add_argument('--batch_size', metavar='-bs', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', metavar='-lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--testing_ratio', metavar='-tr', type=float, default=.1,
                        help='Testing ratio')
    parser.add_argument('--do_figures', metavar='-fig', default=True, action=argparse.BooleanOptionalAction,
                        help='Generate and save figures during local training')

    args = parser.parse_args()

    Split_type = args.scenario
    method = args.method
    std_type = args.standardization
    idx_Test_data = int(args.Test_id)
    tags = args.tags
    data_folder = args.data_folder
    kind = args.kind
    num_samples = args.num_samples
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

    Clients_data, Clients_missing, data_test, data_test_missing, Perc_missing, Perc_missing_test = \
        databases(data_folder,Split_type,idx_clients,idx_Test_data,N_cl,root_dir)
    features = data_test.columns.values.tolist()

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
        fed_mean = fed_mean_std.aggregated_params()[0]['params']['mean']
        fed_std = fed_mean_std.aggregated_params()[0]['params']['std']

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

        standardization = {'type':std_type}
        if method != 'FedProx_loc':
            standardization.update({'fed_mean':fed_mean.tolist(),'fed_std':fed_std.tolist()})

        model_args = {'n_features':data_size, 'n_latent':d,'n_hidden':h,'n_samples':K, 'use_gpu': True, 'n_samples_test':100,
                    'standardization':standardization, 'test_ratio': args.testing_ratio, 'test_on_local_updates': True}

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
        from fedbiomed.researcher.aggregators.fedavg_fedstd import FedAverage_FedStd
        from fedbiomed.researcher.aggregators.scaffold import Scaffold
        from fedbiomed.researcher.aggregators.scaffold_fedstd import Scaffold_FedStd

        if method == 'Scaffold':
            aggregator = Scaffold() if std_type=='static' else Scaffold_FedStd
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

            xmiss, mask, xhat_local_std, xfull_local_std, loc_mean, loc_std =\
                 recover_data(Clients_missing[cls], Clients_data[cls])

            xhat_cls = np.copy(xhat_local_std)
            xhat_0_cls = np.copy(xhat_local_std)
            xfull_cls = np.copy(xfull_local_std)
            mask_cls = np.copy(mask)

            # Model
            encoder_cls, decoder_cls, iota_cls, optimizer_cls = encoder_decoder_iota_opt(p,h,d,lr)

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
                likelihood = (-np.log(K)-miwae_loss(encoder = encoder_cls,decoder = decoder_cls,iota=iota_cls, data = torch.from_numpy(xhat_0_cls).float(),mask = torch.from_numpy(mask_cls).float(), d = d, p = p, K = K).cpu().data.numpy())
                if args.do_figures==True:
                    Loss_cls[cls].append(loss.item())
                    Like_cls[cls].append(likelihood)
                    mse_train = testing_func(features=features,data_missing=xhat_local_std, data_full=xfull_local_std, 
                                             mask=mask, encoder=encoder_cls, decoder=decoder_cls, iota=iota_cls, d=d, L=100)
                    MSE_cls[cls].append(mse_train)
                if ep % rounds == 1:
                    print('Epoch %g' %ep)
                    print('MIWAE likelihood bound  %g' %likelihood) # Gradient step      
                    print('Loss: {:.6f}'.format(loss.item()))

            Encoders_loc.append(encoder_cls)
            Decoders_loc.append(decoder_cls)
            Iota_loc.append(iota_cls)

        # Centralized training
        if args.do_figures==True:
            Loss_tot = []
            Like_tot = []
            MSE_tot = []

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
        encoder_cen, decoder_cen, iota_cen, optimizer_cen = encoder_decoder_iota_opt(p,h,d,lr)

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
                loss = miwae_loss(encoder = encoder_cen,decoder = decoder_cen, iota=iota_cen, data = b_data,mask = b_mask, d = d, p = p, K = K)
                loss.backward()
                optimizer_cen.step()
            likelihood = (-np.log(K)-miwae_loss(encoder = encoder_cen,decoder = decoder_cen,iota=iota_cen, data = torch.from_numpy(xhat_0_tot).float(),mask = torch.from_numpy(mask_tot).float(), d = d, p = p, K = K).cpu().data.numpy())
            if args.do_figures==True:
                Loss_tot.append(loss.item())
                Like_tot.append(likelihood)
                mse_train = testing_func(features=features,data_missing=xhat_0_tot, data_full=xfull_tot, 
                                             mask=mask_tot, encoder=encoder_cen, decoder=decoder_cen, iota=iota_cen, d=d, L=100)
                MSE_tot.append(mse_train)
            if ep % rounds == 1:
                print('Epoch %g' %ep)
                print('MIWAE likelihood bound  %g' %likelihood) # Gradient step      
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
        if std_type=='dynamic':
            fed_mean = model.mean
            fed_std = model.std
    else:
        fed_mean, fed_std = mean_tot_missing, std_tot_missing
        if args.do_figures==True:
            generate_save_plots(result_folder,Loss_cls,Like_cls,MSE_cls,Loss_tot,Like_tot,MSE_tot,n_epochs_local,n_epochs_centralized,idx_clients)

    # Testing on data used during training
    for cls in range(N_cl):
        if ((fed_mean is None) and (fed_std is None)):
            xmiss, mask, xhat_local_std, xfull_local_std, loc_mean, loc_std =\
                recover_data(Clients_missing[cls], Clients_data[cls])
        else:
            xmiss, mask, xhat_global_std, xfull_global_std, xhat_local_std, xfull_local_std, loc_mean, loc_std =\
                recover_data(Clients_missing[cls], Clients_data[cls], fed_mean, fed_std)
        if method != 'Local':
            MSE = testing_func(features=features,data_missing=xhat_local_std, data_full=xfull_local_std, 
                               mask=mask, encoder=encoder, decoder=decoder, iota=iota, d=d, L=L,
                               idx_cl=idx_clients[cls],result_folder=result_folder,method=method+'_local', mean=loc_mean, std=loc_std)
            save_results(result_folder,Split_type,idx_clients,idx_clients[cls],
                Perc_missing,Perc_missing[cls],method,
                N_cl,[len(Clients_missing[i]) for i in range(N_cl)],rounds,n_epochs,
                std_training,'local',MSE)
            if method != 'FedProx_loc':
                MSE = testing_func(features=features,data_missing=xhat_global_std, data_full=xfull_global_std, 
                                   mask=mask, encoder=encoder, decoder=decoder, iota=iota, d=d, L=L,
                                   idx_cl=idx_clients[cls],result_folder=result_folder,method=method+'_global', mean=fed_mean, std=fed_std)
                save_results(result_folder,Split_type,idx_clients,idx_clients[cls],
                    Perc_missing,Perc_missing[cls],method,
                    N_cl,[len(Clients_missing[i]) for i in range(N_cl)],rounds,n_epochs,
                    std_training,'global',MSE)
        elif method == 'Local':
            # centralized 
            MSE = testing_func(features=features,data_missing=xhat_local_std, data_full=xfull_local_std, 
                               mask=mask, encoder=encoder_cen, decoder=decoder_cen, iota=iota_cen, d=d, L=L,
                               idx_cl=idx_clients[cls],result_folder=result_folder,method='Centralized_local', mean=loc_mean, std=loc_std)
            save_results(result_folder,Split_type,sum(idx_clients),idx_clients[cls],
                Perc_missing,Perc_missing[cls],'Centralized',
                1,[len(xmiss_tot)],1,n_epochs_centralized,
                'Loc','local',MSE)
            MSE = testing_func(features=features,data_missing=xhat_global_std, data_full=xfull_global_std, 
                               mask=mask, encoder=encoder_cen, decoder=decoder_cen, iota=iota_cen, d=d, L=L,
                               idx_cl=idx_clients[cls],result_folder=result_folder,method='Centralized_global', mean=fed_mean, std=fed_std)
            save_results(result_folder,Split_type,sum(idx_clients),idx_clients[cls],
                Perc_missing,Perc_missing[cls],'Centralized',
                1,[len(xmiss_tot)],1,n_epochs_centralized,
                'Loc','global',MSE)
            # local
            MSE = testing_func(features=features,data_missing=xhat_local_std, data_full=xfull_local_std, 
                               mask=mask, encoder=Encoders_loc[cls], decoder=Decoders_loc[cls], iota=Iota_loc[cls], d=d, L=L,
                               idx_cl=idx_clients[cls],result_folder=result_folder,method='Local_cl'+str(idx_clients[cls]), mean=loc_mean, std=loc_std)
            save_results(result_folder,Split_type,idx_clients[cls],idx_clients[cls],
                Perc_missing[cls],Perc_missing[cls],'Local_cl'+str(idx_clients[cls]),
                1,[len(Clients_missing[cls])],1,n_epochs_local,
                'Loc','local',MSE)

    # Testing on external dataset
    if ((fed_mean is None) and (fed_std is None)):
        xmiss, mask, xhat_local_std, xfull_local_std, loc_mean, loc_std =\
                recover_data(data_test_missing, data_test)
    else:
        xmiss, mask, xhat_global_std, xfull_global_std, xhat_local_std, xfull_local_std, loc_mean, loc_std =\
                recover_data(data_test_missing, data_test, fed_mean, fed_std)
    if method != 'Local':
        MSE = testing_func(features=features,data_missing=xhat_local_std, data_full=xfull_local_std, 
                           mask=mask, encoder=encoder, decoder=decoder, iota=iota, d=d, L=L,
                           idx_cl=idx_Test_data,result_folder=result_folder,method=method+'_local',kind=kind,
                           num_samples=num_samples, mean=loc_mean, std=loc_std)
        save_results(result_folder,Split_type,idx_clients,idx_Test_data,
            Perc_missing,Perc_missing_test,method,
            N_cl,[len(Clients_missing[i]) for i in range(N_cl)],rounds,n_epochs,
            std_training,'local',MSE)
        if method != 'FedProx_loc':
            MSE = testing_func(features=features,data_missing=xhat_global_std, data_full=xfull_global_std, 
                               mask=mask, encoder=encoder, decoder=decoder, iota=iota, d=d, L=L,
                               idx_cl=idx_Test_data,result_folder=result_folder,method=method+'_global',
                               kind=kind,num_samples=num_samples, mean=fed_mean, std=fed_std)
            save_results(result_folder,Split_type,idx_clients,idx_Test_data,
                Perc_missing,Perc_missing_test,method,
                N_cl,[len(Clients_missing[i]) for i in range(N_cl)],rounds,n_epochs,
                std_training,'global',MSE)
    elif method == 'Local':
        # centralized 
        MSE = testing_func(features=features,data_missing=xhat_local_std, data_full=xfull_local_std, 
                           mask=mask, encoder=encoder_cen, decoder=decoder_cen, iota=iota_cen, d=d, L=L,
                           idx_cl=idx_Test_data,result_folder=result_folder,method='Centralized_local',
                           kind=kind,num_samples=num_samples, mean=loc_mean, std=loc_std)
        save_results(result_folder,Split_type,sum(idx_clients),idx_Test_data,
            Perc_missing,Perc_missing_test,'Centralized',
            1,[len(xmiss_tot)],1,n_epochs_centralized,
            'Loc','local',MSE)
        MSE = testing_func(features=features,data_missing=xhat_global_std, data_full=xfull_global_std, 
                           mask=mask, encoder=encoder_cen, decoder=decoder_cen, iota=iota_cen, d=d, L=L,
                           idx_cl=idx_Test_data,result_folder=result_folder,method='Centralized_global',
                           kind=kind,num_samples=num_samples, mean=fed_mean, std=fed_std)
        save_results(result_folder,Split_type,sum(idx_clients),idx_Test_data,
            Perc_missing,Perc_missing_test,'Centralized',
            1,[len(xmiss_tot)],1,n_epochs_centralized,
            'Loc','global',MSE)
        # local
        for cls in range(N_cl):
            MSE = testing_func(features=features,data_missing=xhat_local_std, data_full=xfull_local_std, 
                               mask=mask, encoder=Encoders_loc[cls], decoder=Decoders_loc[cls], iota=Iota_loc[cls], d=d, L=L,
                               idx_cl=idx_Test_data,result_folder=result_folder,method='Local_cl'+str(idx_clients[cls]),
                               kind=kind,num_samples=num_samples, mean=loc_mean, std=loc_std)
            save_results(result_folder,Split_type,idx_clients[cls],idx_Test_data,
                Perc_missing[cls],Perc_missing_test,'Local_cl'+str(idx_clients[cls]),
                1,[len(Clients_missing[cls])],1,n_epochs_local,
                'Loc','local',MSE)


