import pandas as pd
import numpy as np
import torch
import csv, os
import torch.distributions as td
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

###########################################################
#Define the imputation and the MSE functions              #
###########################################################

def miwae_impute(encoder,decoder,iota,data,x_cat,mask,p,d,L):

    p_z = td.Independent(td.Normal(loc=torch.zeros(d),scale=torch.ones(d)),1)

    batch_size = data.shape[0]
    
    tiledmask = torch.tile(mask,(L,1))
    mask_complement_float = torch.abs(mask-1)

    tilediota = torch.tile(iota,(data.shape[0],1))
    iotax = data + torch.mul(tilediota,mask_complement_float)
    
    ###########################################################
    out_encoder = encoder(torch.cat((iotax,x_cat),dim=1))
    ###########################################################
    
    q_zgivenxobs = td.Independent(td.StudentT(loc=out_encoder[..., :d],\
                                                scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)]),\
                                                df=torch.nn.Softplus()\
                                                (out_encoder[..., (2*d):(3*d)]) + 3),1)

    zgivenx = q_zgivenxobs.rsample([L])
    zgivenx_flat = zgivenx.reshape([L*batch_size,d])

    ###########################################################
    out_decoder = decoder(torch.cat((zgivenx_flat,x_cat.repeat(L,1)),dim=1))
    ###########################################################

    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
    all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3

    data_flat = torch.Tensor.repeat(data,[L,1]).reshape([-1,1])
    #tiledmask = torch.Tensor.repeat(mask,[L,1])

    all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),\
            scale=all_scales_obs_model.reshape([-1,1]),\
            df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,p])

    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)

    imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
    xms = xgivenz.mean.reshape([L,batch_size,p])  # that's the only line that changed!
    xm=torch.einsum('ki,kij->ij', imp_weights, xms) 

    return xm

def mse(xhat,xtrue,mask): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.mean(np.power(xhat-xtrue,2)[~mask])

def miwae_loss(encoder, decoder, iota, data, xcat, mask, d, p, K):
    p_z = td.Independent(td.Normal(loc=torch.zeros(d),scale=torch.ones(d)),1)
    batch_size = data.shape[0]
        
    tiledmask = torch.tile(mask,(K,1))
    mask_complement_float = torch.abs(mask-1)

    tilediota = torch.tile(iota,(data.shape[0],1))
    iotax = data + torch.mul(tilediota,mask_complement_float)
    
    ###########################################################
    out_encoder = encoder(torch.cat((iotax,xcat),dim=1))
    ###########################################################
    
    q_zgivenxobs = td.Independent(td.StudentT(loc=out_encoder[..., :d],\
                                            scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)]),\
                                            df=torch.nn.Softplus()\
                                            (out_encoder[..., (2*d):(3*d)]) + 3),1)

    zgivenx = q_zgivenxobs.rsample([K])
    zgivenx_flat = zgivenx.reshape([K*batch_size,d])

    ###########################################################
    out_decoder = decoder(torch.cat((zgivenx_flat,xcat.repeat(K,1)),dim=1))
    ###########################################################

    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
    all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3

    data_flat = torch.Tensor.repeat(data,[K,1]).reshape([-1,1])
    #tiledmask = torch.Tensor.repeat(mask,[K,1])

    all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),
                            scale=all_scales_obs_model.reshape([-1,1]),
                            df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,p])

    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))

    return neg_bound

def recover_data(data_missing, data_full, n_cov, fed_mean = None, fed_std = None):

    # TEST DATA: data_missing = data_test_missing; data_full = data_test
    # CLIENT i, i=0,1,2: data_missing = Clients_missing[i]; data_full = Clients_data[i]

    xmiss = np.copy(data_missing)
    xmiss_cont = xmiss[:,n_cov:]
    
    x_cov = xmiss[:,:n_cov]

    xfull = np.copy(data_full)
    xfull_cont = xfull[:,n_cov:]

    xmiss_cont_local = standardize_data(xmiss_cont)
    xhat_local_std = np.concatenate((x_cov, xmiss_cont_local), axis=1)
    xhat_local_std[np.isnan(xhat_local_std)] = 0

    xfull_cont_local = standardize_data(xfull_cont)
    xfull_local_std = np.concatenate((x_cov, xfull_cont_local), axis=1)

    if ((fed_mean is not None) and (fed_std is not None)): 
        if ((type(fed_mean) != np.ndarray) and (type(fed_std) != np.ndarray)):
            fed_mean, fed_std = fed_mean.numpy(), fed_std.numpy()
            
        # standardization with respect to the federated dataset
        xmiss_cont_global = standardize_data(xmiss_cont, fed_mean, fed_std)
        xhat_global_std = np.concatenate((x_cov, xmiss_cont_global), axis=1)
        xhat_global_std[np.isnan(xhat_global_std)] = 0

        xfull_cont_global = standardize_data(xfull_cont, fed_mean, fed_std)
        xfull_global_std = np.concatenate((x_cov, xfull_cont_global), axis=1)

        return xhat_global_std, xfull_global_std, xhat_local_std, xfull_local_std
    else:
        return xhat_local_std, xfull_local_std
        
def standardize_data(data, fed_mean = None, fed_std = None):
    data_norm = np.copy(data)
    if ((fed_mean is not None) and (fed_std is not None)):
        data_norm = (data_norm - fed_mean)/fed_std
    else:
        data_norm = (data_norm - np.nanmean(data_norm,0))/np.nanstd(data_norm,0)
    return data_norm

def testing_func(data_missing, data_full, x_cat, mask, encoder, decoder, iota, d,L):

    xhat = np.copy(data_missing)
    xhat_0 = np.copy(data_missing)
    xfull = np.copy(data_full)
    x_cat = np.copy(x_cat)

    p = data_full.shape[1] # number of features

    xhat[~mask] = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, data = torch.from_numpy(xhat_0).float(), x_cat = torch.from_numpy(x_cat).float(),
                            mask = torch.from_numpy(mask).float(),p=p, d = d,L= L).cpu().data.numpy()[~mask]
    err = np.array([mse(xhat,xfull,mask)])

    return float(err)

def save_results_imputation(result_folder, Train_data,Test_data,model,
                N_train_centers,Size,N_rounds,N_epochs,
                std_training,std_testing,MSE):

    os.makedirs(result_folder, exist_ok=True) 
    exp_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output_file_name = 'output_'+str(exp_id)+'_'+str(np.random.randint(9999, dtype=int))+'.csv'
    fieldnames=['Train_data', 'Test_data', 'model', 
                'N_train_centers', 'Size', 'N_rounds', 'N_epochs',
                'std_training', 'std_testing', 'MSE']
    if not os.path.exists(result_folder+'/'+output_file_name):
        output = open(result_folder+'/'+output_file_name, "w")
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter = ';')
        writer.writeheader()
        output.close()

    # Dictionary to be added
    dict_out={'Train_data': Train_data, 'Test_data': Test_data, 
                'model': model, 'N_train_centers': N_train_centers, 'Size': Size, 
                'N_rounds': N_rounds, 'N_epochs': N_epochs,
                'std_training': std_training, 'std_testing': std_testing, 'MSE': MSE}

    with open(result_folder+'/'+output_file_name, 'a') as output_file:
        dictwriter_object = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter = ';')
        dictwriter_object.writerow(dict_out)
        output_file.close()

def databases(data_folder,task,idx_clients,idx_Test_data,root_dir=None):

    if task == 'imputation':
        data_folder_miss = data_folder + 'clients_missing'
        test_missing_file = os.path.join(str(data_folder_miss),"dataset_"+str(idx_Test_data)+".csv")
        data_test_missing = pd.read_csv(test_missing_file, sep=",",index_col=False)
        test_mask_file = os.path.join(str(data_folder_miss),"mask_"+str(idx_Test_data)+".csv")
        test_mask = pd.read_csv(test_mask_file, sep=",",index_col=False)
        test_mask_np = np.copy(test_mask)
        test_mask_np = np.invert(test_mask_np) 
        Clients_missing=[]
        Clients_mask = []
    elif task == 'prediction':
        test_label_file = os.path.join(str(data_folder),"Labels_"+str(idx_Test_data)+".csv")
        test_label = pd.read_csv(test_label_file, sep=",",index_col=False)
        Clients_label = []
    
    data_folder += 'clients'

    if root_dir is not None:
        root_dir = Path.home() if root_dir == 'home' else Path.home().joinpath( 'Documents/INRIA_EPIONE/FedBioMed', 'fedbiomed' )
        data_folder = root_dir.joinpath(data_folder)

    Clients_data=[]
    for i in idx_clients:
        if task == 'imputation':
            data_full_file = os.path.join(str(data_folder), "dataset_full_"+str(i)+".csv")
            data_full = pd.read_csv(data_full_file, sep=",",index_col=False)
            Clients_data.append(data_full)
            data_file = os.path.join(str(data_folder_miss),"dataset_"+str(i)+".csv")
            data = pd.read_csv(data_file, sep=",",index_col=False)
            Clients_missing.append(data)
            mask_file = os.path.join(str(data_folder_miss),"mask_"+str(i)+".csv")
            mask = pd.read_csv(mask_file, sep=",",index_col=False)
            mask_np = np.copy(mask)
            mask_np = np.invert(mask_np) 
            Clients_mask.append(mask_np)
        elif task == 'prediction':
            data_full_file = os.path.join(str(data_folder), "Client_"+str(i)+".csv")
            data_full = pd.read_csv(data_full_file, sep=",",index_col=False)
            Clients_data.append(data_full)
            label_file = os.path.join(str(data_folder),"Labels_"+str(i)+".csv")
            label = pd.read_csv(label_file, sep=",",index_col=False)
            Clients_label.append(label)            

    test_file = os.path.join(str(data_folder),"dataset_full_"+str(idx_Test_data)+".csv")
    data_test = pd.read_csv(test_file, sep=",",index_col=False)

    if task == 'imputation':
        return Clients_data, Clients_missing, Clients_mask, data_test, data_test_missing, test_mask_np
    elif task == 'prediction':
        return Clients_data, Clients_label, data_test, test_label

def generate_save_plots(result_folder,Loss_cls,Like_cls,MSE_cls,Loss_tot,Like_tot,MSE_tot,epochs_loc,epochs_tot,idx_clients):
    figures_folder = result_folder+'/Figures'
    os.makedirs(figures_folder, exist_ok=True)
    exp_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'_'+str(np.random.randint(9999, dtype=int))
    for cl in range(len(idx_clients)):
        file_name = exp_id+'Fig_client_'+str(idx_clients[cl])
        save_plots(epochs_loc,Like_cls[cl],Loss_cls[cl],MSE_cls[cl],figures_folder,file_name)
    file_name = exp_id+'Fig_centralized_'+str(idx_clients)
    save_plots(epochs_tot,Like_tot,Loss_tot,MSE_tot,figures_folder,file_name)

def save_plots(epochs,likelihood,loss,mse,fig_folder,file_name):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax0.plot(range(1,epochs), likelihood)
    ax0.set(ylabel='MIWAE likelihood bound')
    ax1.plot(range(1,epochs), loss)
    ax1.set(ylabel='Loss')
    ax2.plot(range(1,epochs), mse)
    ax2.set(xlabel='Epochs',ylabel='MSE')
    plt.savefig(fig_folder + '/' + file_name)
    plt.clf()
    plt.close()