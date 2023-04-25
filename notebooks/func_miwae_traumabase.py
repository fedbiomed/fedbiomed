import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import csv, os
import torch.distributions as td
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.lines import Line2D
import seaborn as sns; sns.set()

###########################################################
#Define the imputation and the MSE functions              #
###########################################################

def encoder_decoder_iota_opt(p,num_covariates,h,d,lr):
    encoder = nn.Sequential(
        torch.nn.Linear(p+num_covariates, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, 3*d),  
    )

    decoder = nn.Sequential(
        torch.nn.Linear(d+num_covariates, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, 3*p),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
    )

    iota = nn.Parameter(torch.zeros(1,p),requires_grad=True)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + [iota],lr=lr)

    def weights_init(layer):
        if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
            
    encoder.apply(weights_init)
    decoder.apply(weights_init)

    return encoder, decoder, iota, optimizer

def miwae_impute(encoder,decoder,iota,data,x_cat,mask,p,d,L,kind="single",num_samples=20):
    with torch.no_grad():

        # kind = "single" will return only the single imputation
        # kind = "multiple" will return num_samples multiple imputations
        # kind = "both" will both the single imputation and num_samples multiple imputations

        p_z = td.Independent(td.Normal(loc=torch.zeros(d),scale=torch.ones(d)),1)

        batch_size = data.shape[0]

        if np.isnan(data).any():
            data[np.isnan(data)] = 0
        
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
        log_imp_weights = logpxobsgivenz + logpz - logq # same importance wieghts used for single and multiple imputation

        imp_weights = torch.nn.functional.softmax(log_imp_weights,0) # these are w_1,....,w_L for all observations in the batch
        xms = xgivenz.mean.reshape([L,batch_size,p])  # that's the only line that changed!
        xm = torch.multiply(torch.einsum('ki,kij->ij', imp_weights, xms),mask_complement_float)

        if ((kind=="multiple") or (kind=="both")):

            sir_logits = torch.t(log_imp_weights)
            sir = td.Categorical(logits = sir_logits).sample((num_samples,))
            xmul = torch.reshape(torch.multiply(torch.reshape(xgivenz.sample(),[L,batch_size,p]),mask_complement_float),[L,batch_size,p])
            xmul_gat = xmul.permute(1,0,2)[:,torch.t(sir)][0]

    if (kind=="single"):
        return xm

    else:
        return xm, xmul_gat

# def miwae_impute(encoder,decoder,iota,data,x_cat,mask,p,d,L):

#     p_z = td.Independent(td.Normal(loc=torch.zeros(d),scale=torch.ones(d)),1)

#     batch_size = data.shape[0]

#     if np.isnan(data).any():
#         data[np.isnan(data)] = 0
    
#     tiledmask = torch.tile(mask,(L,1))
#     mask_complement_float = torch.abs(mask-1)

#     tilediota = torch.tile(iota,(data.shape[0],1))
#     iotax = data + torch.mul(tilediota,mask_complement_float)
    
#     ###########################################################
#     out_encoder = encoder(torch.cat((iotax,x_cat),dim=1))
#     ###########################################################
    
#     q_zgivenxobs = td.Independent(td.StudentT(loc=out_encoder[..., :d],\
#                                                 scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)]),\
#                                                 df=torch.nn.Softplus()\
#                                                 (out_encoder[..., (2*d):(3*d)]) + 3),1)

#     zgivenx = q_zgivenxobs.rsample([L])
#     zgivenx_flat = zgivenx.reshape([L*batch_size,d])

#     ###########################################################
#     out_decoder = decoder(torch.cat((zgivenx_flat,x_cat.repeat(L,1)),dim=1))
#     ###########################################################

#     all_means_obs_model = out_decoder[..., :p]
#     all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
#     all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3

#     data_flat = torch.Tensor.repeat(data,[L,1]).reshape([-1,1])
#     #tiledmask = torch.Tensor.repeat(mask,[L,1])

#     all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),\
#             scale=all_scales_obs_model.reshape([-1,1]),\
#             df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
#     all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,p])

#     logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
#     logpz = p_z.log_prob(zgivenx)
#     logq = q_zgivenxobs.log_prob(zgivenx)

#     xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)

#     imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
#     xms = xgivenz.mean.reshape([L,batch_size,p])  # that's the only line that changed!
#     xm=torch.einsum('ki,kij->ij', imp_weights, xms) 

#     return xm

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

def mean_std_filled(data):
    #Fill nan values with mean
    data_filled = np.copy(data)
    col_mean = np.nanmean(data_filled,0)
    #Find indices that you need to replace
    inds = np.where(np.isnan(data_filled))
    #Place column means in the indices. Align the arrays using take
    data_filled[inds] = np.take(col_mean, inds[1])
    mean,std = np.nanmean(data_filled,0), np.nanstd(data_filled,0)
    return mean,std

def recover_data(data_missing, data_full, n_cov, fed_mean = None, fed_std = None):

    # TEST DATA: data_missing = data_test_missing; data_full = data_test
    # CLIENT i, i=0,1,2: data_missing = Clients_missing[i]; data_full = Clients_data[i]

    xmiss = np.copy(data_missing)
    xmiss_cont = xmiss[:,n_cov:]
    
    x_cov = xmiss[:,:n_cov]

    xfull = np.copy(data_full)
    xfull_cont = xfull[:,n_cov:]

    ###########################################################
    #mean,std = mean_std_filled(xmiss_cont)
    #xmiss_cont_local = standardize_data(xmiss_cont,mean,std)
    ###########################################################

    xmiss_cont_local = standardize_data(xmiss_cont)
    xhat_local_std = np.concatenate((x_cov, xmiss_cont_local), axis=1)
    xhat_local_std[np.isnan(xhat_local_std)] = 0

    ###########################################################
    #mean,std = mean_std_filled(xfull_cont)
    #xfull_cont_local = standardize_data(xfull_cont,mean,std)
    ###########################################################
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

def create_save_data_prediction(encoder, decoder, iota, data, n_cov, result_folder, filename, d, L, 
                                standard = False, mean = None, std = None,kind="single",num_samples=20):
    
    cwd = os.getcwd()
    folder = cwd+'/'+result_folder+'/clients_imputed'
    os.makedirs(folder, exist_ok=True)
    
    col_names = list(data.columns)
    data_np = np.copy(data)
    mask = np.isfinite(data_np)[:,n_cov:]
    data_cont = data_np[:,n_cov:] 
    data_cov = data_np[:,:n_cov]
    p = len(col_names)-n_cov
    n = data.shape[0] # number of features

    if standard:
        if ((mean is None) and (std is None)):
            ###########################################################
            #mean,std = mean_std_filled(data_cont)
            mean, std = np.nanmean(data_cont,0), np.nanstd(data_cont,0)
            ###########################################################
        else:    
            if ((type(mean) != np.ndarray) and (type(std) != np.ndarray)):
                mean, std = mean.numpy(), std.numpy()
            #mean, std = mean[1:], std[1:]
        data_cont = standardize_data(data_cont, mean, std)

    # data_cont[~mask] = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, data = torch.from_numpy(data_cont).float(), 
    #                         x_cat = torch.from_numpy(data_cov).float(), mask = torch.from_numpy(mask).float(),p=p, d = d, L= L).cpu().data.numpy()[~mask]

    for i in range(n):
        if (np.sum(mask[i,:])<=p-1):
            data_cont[i,~mask[i,:]] = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, data = torch.from_numpy(data_cont[i,:]).float().reshape([1,p]), 
                                          x_cat = torch.from_numpy(data_cov[i,:]).float().reshape([1,n_cov]),
                                          mask = torch.from_numpy(mask[i,:]).float().reshape([1,p]),p=p, d = d,L= L).cpu().data.numpy()[0][~mask[i,:]]

    if standard:
        data_cont = data_cont*std + mean
    data_cont = np.round(data_cont,1)
    data_np = np.concatenate((data_cov, data_cont), axis=1)

    data_df = pd.DataFrame(data_np, columns=col_names)

    data_df.to_csv(folder+'/'+filename+'single.csv',index=False)
    
    if (kind!="single"):
        data_cont_i = np.copy(data)[:,n_cov:]
        if standard:
            if ((mean is None) and (std is None)):
                mean, std = np.nanmean(data_cont_i,0), np.nanstd(data_cont_i,0)
            else:    
                if ((type(mean) != np.ndarray) and (type(std) != np.ndarray)):
                    mean, std = mean.numpy(), std.numpy()
            data_cont_i = standardize_data(data_cont_i, mean, std)
        Data_mul=[]
        for _ in range(num_samples):
            data_imp_mul_i = np.copy(data_cont_i)
            Data_mul.append(data_imp_mul_i)
        # Data_mul = [data_cont_i for _ in range(num_samples)]
        for i in range(n):
            if (np.sum(mask[i,:])<=p-1):
                xhat_single, xhat_multiple = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, 
                                                        data = torch.from_numpy(data_cont_i[i,:]).reshape([1,p]).float(), 
                                                        x_cat = torch.from_numpy(data_cov[i,:]).float().reshape([1,n_cov]), 
                                                        mask = torch.from_numpy(mask[i,:]).reshape([1,p]).float(),
                                                        p=p, d = d,L= L,kind=kind,num_samples=num_samples)
                for sam in range(num_samples):
                    Data_mul[sam][i,~mask[i,:]] = xhat_multiple.numpy()[0,sam,~mask[i,:]]
                    print(xhat_multiple.numpy()[0,sam,~mask[i,:]])

        for sam in range(num_samples):
            if standard:
                Data_mul[sam] = Data_mul[sam]*std + mean
            Data_mul[sam] = np.round(Data_mul[sam],1)
            data_np_imp = np.concatenate((data_cov, Data_mul[sam]), axis=1)

            data_df = pd.DataFrame(data_np_imp, columns=col_names)

            data_df.to_csv(folder+'/'+filename+'multiple'+str(sam)+'.csv',index=False)

def recover_data_prediction(data_full , n_cov, fed_mean = None, fed_std = None):

    xfull = np.copy(data_full)
    x_cov = xfull[:,:n_cov]
    xfull_cont = xfull[:,n_cov:]

    ###########################################################
    #mean,std = mean_std_filled(xfull_cont)
    #xfull_cont_local = standardize_data(xfull_cont,mean,std)
    ###########################################################

    xfull_cont_local = standardize_data(xfull_cont)
    xfull_local_std = np.concatenate((x_cov, xfull_cont_local), axis=1)

    if ((fed_mean is not None) and (fed_std is not None)): 
        if ((type(fed_mean) != np.ndarray) and (type(fed_std) != np.ndarray)):
            fed_mean, fed_std = fed_mean.numpy(), fed_std.numpy()
        xfull_cont_global = standardize_data(xfull_cont, fed_mean, fed_std)
        xfull_global_std = np.concatenate((x_cov, xfull_cont_global), axis=1)

        return xfull_global_std, xfull_local_std
    else:
        return xfull_local_std

        
def standardize_data(data, fed_mean = None, fed_std = None):
    data_norm = np.copy(data)
    if ((fed_mean is not None) and (fed_std is not None)):
        data_norm = (data_norm - fed_mean)/fed_std
    else:
        data_norm = (data_norm - np.nanmean(data_norm,0))/np.nanstd(data_norm,0)
    return data_norm

def testing_func_mul(features, data_missing, data_full, x_cat, mask, encoder, decoder, iota, d,L,
                 idx_cl=4,result_folder='results',method='FedAvg',kind="single",num_samples=20):

    #features = data_full.columns.values.tolist()
    xhat = np.copy(data_missing)
    xhat_0 = np.copy(data_missing)
    xfull = np.copy(data_full)
    x_cat = np.copy(x_cat)

    p = data_full.shape[1] # number of features
    n = data_full.shape[0] # number of features
    ncov = x_cat.shape[1]

    #xhat[~mask] = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, data = torch.from_numpy(xhat_0).float(), x_cat = torch.from_numpy(x_cat).float(),
    #                    mask = torch.from_numpy(mask).float(),p=p, d = d,L= L).cpu().data.numpy()[~mask]

    for i in range(n):
        if (np.sum(mask[i,:])<=p-1):
            xhat[i,~mask[i,:]] = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, data = torch.from_numpy(xhat_0[i,:]).float().reshape([1,p]), x_cat = torch.from_numpy(x_cat[i,:]).float().reshape([1,ncov]),
                                mask = torch.from_numpy(mask[i,:]).float().reshape([1,p]),p=p, d = d,L= L).cpu().data.numpy()[0][~mask[i,:]]
    err = np.array([mse(xhat,xfull,mask)])

    #xm = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, data = torch.from_numpy(xhat_0).float(),
    #                           mask = torch.from_numpy(mask).float(),p=p, d = d,L= L)
    #    
    #xhat[~mask] = xm.cpu().data.numpy()[~mask]
    #err = np.array([mse(xhat,xfull,mask)])

    if (kind=="single"):
        return float(err)
    else:
        for i in range(n):
            if (np.sum(mask[i,:])<=p-2):
                xhat_single, xhat_multiple = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, 
                                                        data = torch.from_numpy(xhat_0[i,:]).reshape([1,p]).float(), 
                                                        mask = torch.from_numpy(mask[i,:]).reshape([1,p]).float(),
                                                        p=p, d = d,L= L,kind=kind,num_samples=num_samples)


                true_values = xfull[i,~mask[i,:].astype(bool)]
                single_imp = np.squeeze(xhat_single[:,~mask[i,:].astype(bool)])
                mul_imp = np.squeeze(xhat_multiple.numpy()[:,:,~mask[i,:].astype(bool)])
                features_i = np.array(features)[~mask[i,:].astype(bool)]
                multiple_imputation_plot(result_folder,xfull[:,~mask[i,:].astype(bool)],mul_imp,single_imp,true_values,method,idx_cl,i,features_i)

        return float(err)

def testing_func(data_missing, data_full, x_cat, mask, encoder, decoder, iota, d,L):

    xhat = np.copy(data_missing)
    xhat_0 = np.copy(data_missing)
    xfull = np.copy(data_full)
    x_cat = np.copy(x_cat)

    p = data_full.shape[1] # number of features
    n = data_full.shape[0] # number of features
    ncov = x_cat.shape[1]

    for i in range(n):
        if (np.sum(mask[i,:])<=p-1):
    #xhat[~mask] = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, data = torch.from_numpy(xhat_0).float(), x_cat = torch.from_numpy(x_cat).float(),
    #                    mask = torch.from_numpy(mask).float(),p=p, d = d,L= L).cpu().data.numpy()[~mask]
            xhat[i,~mask[i,:]] = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, data = torch.from_numpy(xhat_0[i,:]).float().reshape([1,p]), x_cat = torch.from_numpy(x_cat[i,:]).float().reshape([1,ncov]),
                                mask = torch.from_numpy(mask[i,:]).float().reshape([1,p]),p=p, d = d,L= L).cpu().data.numpy()[0][~mask[i,:]]
    err = np.array([mse(xhat,xfull,mask)])
    #for var in range(p):
    #    err_var = np.array([mse(xhat[ :,var],xfull[ :,var],mask[ :,var])])
    #    print(err_var)
    #print(np.isnan(xfull[~mask]).any(), xfull[~mask].shape)
    #print(err)
    #print(xfull[~mask])

    return float(err)

def save_results_imputation(result_folder, mask_num, Train_data,Test_data,model,
                N_train_centers,Size,N_rounds,N_epochs,
                std_training,std_testing,MSE):

    os.makedirs(result_folder, exist_ok=True) 
    exp_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output_file_name = 'output_'+str(exp_id)+'_'+str(np.random.randint(9999, dtype=int))+'.csv'
    fieldnames=['mask_num','Train_data', 'Test_data', 'model', 
                'N_train_centers', 'Size', 'N_rounds', 'N_epochs',
                'std_training', 'std_testing', 'MSE']
    if not os.path.exists(result_folder+'/'+output_file_name):
        output = open(result_folder+'/'+output_file_name, "w")
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter = ';')
        writer.writeheader()
        output.close()

    # Dictionary to be added
    dict_out={'mask_num': mask_num, 'Train_data': Train_data, 'Test_data': Test_data, 
                'model': model, 'N_train_centers': N_train_centers, 'Size': Size, 
                'N_rounds': N_rounds, 'N_epochs': N_epochs,
                'std_training': std_training, 'std_testing': std_testing, 'MSE': MSE}

    with open(result_folder+'/'+output_file_name, 'a') as output_file:
        dictwriter_object = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter = ';')
        dictwriter_object.writerow(dict_out)
        output_file.close()

def save_model(result_folder,regressor,coef_,intercept_):
    os.makedirs(result_folder, exist_ok=True) 
    torch.save(coef_, f'{result_folder}/{regressor}_trained_model_coef')
    torch.save(intercept_, f'{result_folder}/{regressor}_trained_model_intercept')

def save_results_prediction(result_folder, model, regressor, Epochs_reg, Rounds_reg, F1, 
                            precision, mse, accuracy, conf_matr, validation_err,
                            coefs, feat_names):
    os.makedirs(result_folder, exist_ok=True) 
    exp_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output_file_name = 'output_pred_'+str(exp_id)+'_'+str(np.random.randint(9999, dtype=int))+'.csv'
    fieldnames=['model', 'regressor', 'N_rounds_reg', 'N_epochs_reg',
                'F1', 'precision', 'mse', 'accuracy', 'conf_matr', 'validation_err',
                'coefficients','features names']
    if not os.path.exists(result_folder+'/'+output_file_name):
        output = open(result_folder+'/'+output_file_name, "w")
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter = ';')
        writer.writeheader()
        output.close()

    # Dictionary to be added
    dict_out={'model': model, 'regressor': regressor, 'N_rounds_reg': Rounds_reg, 'N_epochs_reg': Epochs_reg, 
                'F1':F1, 'precision':precision, 'mse':mse, 'accuracy':accuracy, 
                'conf_matr':conf_matr, 'validation_err':validation_err,
                'coefficients': coefs,'features names': feat_names}

    with open(result_folder+'/'+output_file_name, 'a') as output_file:
        dictwriter_object = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter = ';')
        dictwriter_object.writerow(dict_out)
        output_file.close()

def databases(data_folder,task,idx_clients,mask_num,root_dir=None,idx_Test_data=None, inputed = False):

    data_folder_miss = data_folder + 'clients_missing_' + str(mask_num)
    data_folder += 'clients'
    if root_dir is not None:
        root_dir = Path.home() if root_dir == 'home' else Path.home().joinpath( 'Documents/INRIA_EPIONE/FedBioMed', 'fedbiomed' )
        data_folder = root_dir.joinpath(data_folder)

    if task == 'imputation':
        test_missing_file = os.path.join(str(data_folder_miss),"dataset_"+str(idx_Test_data)+".csv")
        data_test_missing = pd.read_csv(test_missing_file, sep=",",index_col=False)
        if idx_Test_data is not None:
            test_mask_file = os.path.join(str(data_folder_miss),"mask_"+str(idx_Test_data)+".csv")
            test_mask = pd.read_csv(test_mask_file, sep=",",index_col=False)
            test_mask_np = np.copy(test_mask)
            test_mask_np = np.invert(test_mask_np) 
        Clients_missing=[]
        Clients_mask = []
    elif task == 'prediction':
        if idx_Test_data is not None:
            test_label_file = os.path.join(str(data_folder),"Labels_"+str(idx_Test_data)+".csv")
            test_label = pd.read_csv(test_label_file, sep=",",index_col=False)
        Clients_label = []

    Clients_data=[]
    for i in idx_clients:
        if task == 'imputation':
            data_full_file = os.path.join(str(data_folder), "dataset_full_"+str(i)+".csv")
            data_file = os.path.join(str(data_folder_miss),"dataset_"+str(i)+".csv")
            data = pd.read_csv(data_file, sep=",",index_col=False)
            Clients_missing.append(data)
            mask_file = os.path.join(str(data_folder_miss),"mask_"+str(i)+".csv")
            mask = pd.read_csv(mask_file, sep=",",index_col=False)
            mask_np = np.copy(mask)
            mask_np = np.invert(mask_np) 
            Clients_mask.append(mask_np)
        elif task == 'prediction':
            if inputed:
                data_full_file = os.path.join(str(data_folder), "Client_inputed_"+str(i)+".csv") 
            else:   
                data_full_file = os.path.join(str(data_folder), "Client_"+str(i)+".csv")
            label_file = os.path.join(str(data_folder),"Labels_"+str(i)+".csv")
            label = pd.read_csv(label_file, sep=",",index_col=False)
            Clients_label.append(label)            
        data_full = pd.read_csv(data_full_file, sep=",",index_col=False)
        Clients_data.append(data_full)

    if idx_Test_data is not None:
        if task == 'imputation':
            test_file = os.path.join(str(data_folder),"dataset_full_"+str(idx_Test_data)+".csv")
        elif task == 'prediction':
            if inputed:
                test_file = os.path.join(str(data_folder),"Client_inputed_"+str(idx_Test_data)+".csv")
            else:
                test_file = os.path.join(str(data_folder),"Client_"+str(idx_Test_data)+".csv")
        data_test = pd.read_csv(test_file, sep=",",index_col=False)

    if task == 'imputation':
        if idx_Test_data is not None:
            return Clients_data, Clients_missing, Clients_mask, data_test, data_test_missing, test_mask_np
        else:
            return Clients_data, Clients_missing, Clients_mask
    elif task == 'prediction':
        if idx_Test_data is not None:
            return Clients_data, Clients_label, data_test, test_label
        else:
            return Clients_data, Clients_label
        
def load_and_predict_data(sam,data_folder,idx_Test_data,root_dir=None):
    data_folder += 'clients_imputed'
    if root_dir is not None:
        root_dir = Path.home() if root_dir == 'home' else Path.home().joinpath( 'Documents/INRIA_EPIONE/FedBioMed', 'fedbiomed' )
        data_folder = root_dir.joinpath(data_folder)
    test_file = os.path.join(str(data_folder),"Client_imputed_"+str(idx_Test_data)+"multiple"+str(sam)+".csv")
    data_test = pd.read_csv(test_file, sep=",",index_col=False)
    return data_test

def save_results_load_and_predict(result_folder,Prediction_df):
    Prediction_df.to_csv(result_folder+'/prediction_multiple_imp.csv',index=False)

def databases_pred(data_folder,idx_clients,
                   root_dir=None, idx_Test_data=None, imputed = False):
    if imputed:
        data_folder += 'clients_imputed'
    else:
        data_folder += 'clients'
    if root_dir is not None:
        root_dir = Path.home() if root_dir == 'home' else Path.home().joinpath( 'Documents/INRIA_EPIONE/FedBioMed', 'fedbiomed' )
        data_folder = root_dir.joinpath(data_folder)

    Clients_data=[]
    for i in idx_clients:
        if imputed:
                data_full_file = os.path.join(str(data_folder), "Client_imputed_"+str(i)+"single.csv") 
        else:   
            data_full_file = os.path.join(str(data_folder), "Client_"+str(i)+".csv")
        data_full = pd.read_csv(data_full_file, sep=",",index_col=False)
        Clients_data.append(data_full)
    if imputed:
        test_file = os.path.join(str(data_folder),"Client_imputed_"+str(idx_Test_data)+"single.csv")
    else:
        test_file = os.path.join(str(data_folder),"Client_"+str(idx_Test_data)+".csv")
    data_test = pd.read_csv(test_file, sep=",",index_col=False)
    return Clients_data, data_test

def generate_save_plots(result_folder,Loss,Like,MSE,epochs,model,idx_clients):
    figures_folder = result_folder+'/Figures'
    os.makedirs(figures_folder, exist_ok=True)
    exp_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'_'+str(np.random.randint(9999, dtype=int))
    if model == 'Local':
        for cl in range(len(idx_clients)):
            file_name = exp_id+'Fig_client_'+str(idx_clients[cl])
            save_plots(epochs,Like[cl],Loss[cl],MSE[cl],figures_folder,file_name)
    elif model == 'Centralized':
        file_name = exp_id+'Fig_centralized_'+str(idx_clients)
        save_plots(epochs,Like,Loss,MSE,figures_folder,file_name)

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

def generate_save_plots_prediction(result_folder,testing_error,validation_error,conf_matr,model,regressor):
    figures_folder = result_folder+'/Figures_pred'
    os.makedirs(figures_folder, exist_ok=True)
    exp_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'_'+str(np.random.randint(9999, dtype=int))
    
    plt.plot(testing_error)
    plt.title('FL testing loss')
    plt.xlabel('FL round')
    plt.ylabel('testing loss (MSE)')
    filename = exp_id+'_Testing_error_'+model+'_'+regressor
    plt.savefig(figures_folder + '/' + filename)
    plt.clf()
    plt.close()

    plt.plot(validation_error)
    plt.title('FL validation err')
    plt.xlabel('FL round')
    plt.ylabel('Validation error')
    filename = exp_id+'_Validation_error_'+model+'_'+regressor
    plt.savefig(figures_folder + '/' + filename)
    plt.clf()
    plt.close()

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matr)
    disp.plot()
    filename = exp_id+'_Conf_matrix_'+model+'_'+regressor
    plt.savefig(figures_folder + '/' + filename)
    plt.clf()
    plt.close()

    #plt.scatter(y_pred, y_test, label='model prediction')
    #plt.xlabel('predicted')
    #plt.ylabel('target')
    #plt.title('Federated model testing prediction')
    #
    #first_diag = np.arange(np.min(y_test.flatten()),
    #                    np.max(y_test.flatten()+1))
    #plt.scatter(first_diag, first_diag, label='correct Target')
    #plt.legend()    
    #filename = exp_id+'_Model_prediction_'+model
    #plt.savefig(figures_folder + '/' + filename)
    #plt.clf()
    #plt.close()

def multiple_imputation_plot(result_folder,xfull,mul_imp,single_imp,true_values,method,idx_cl,i,features_i):
    figures_folder = result_folder+'/Figures'
    os.makedirs(figures_folder, exist_ok=True)
    exp_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'_'+str(np.random.randint(9999, dtype=int))
    file_name = 'Multiple_imp_'+method+exp_id+'Client_'+str(idx_cl)
    ax = plt.figure(figsize=(9,10))
    sns.kdeplot(x=mul_imp[:,0],y=mul_imp[:,1],fill=True)
    plt.scatter(x=single_imp[0],y=single_imp[1], s =100, marker = "P",  c = '#2ca02c')
    plt.scatter(x=true_values[0],y=true_values[1], s =100, marker = "X",  c = '#d62728')
    legend_elements = [Line2D([0], [0], color='b', lw=4, label='MIWAE (KDE of multiple imp.)'),
                    Line2D([0], [0], marker='P', color='w', markerfacecolor = '#2ca02c', label='MIWAE (single imp.)', markersize=20, lw=0),
                    Line2D([0], [0], marker='X', color='w', markerfacecolor = '#d62728', label='True value', markersize=20, lw=0)]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=25)
    plt.xlabel(str(features_i[0]), fontsize=25)
    plt.ylabel(str(features_i[1]), fontsize=25)
    plt.tick_params(axis='both', labelsize = 20)
    plt.title('Subject'+str(i))
    plt.savefig(figures_folder + '/' + file_name +'.pdf',format='pdf',bbox_inches='tight')
    plt.clf()
    plt.close()

    ax = plt.figure(figsize=(9,10))
    file_name = 'px_miss_px_obs_'+method+exp_id+'Client_'+str(idx_cl)
    sns.kdeplot(x=mul_imp[:,0],y=mul_imp[:,1], color = 'b')
    sns.kdeplot(x=xfull[:,0],y=xfull[:,1], color = 'r')
    legend_elements = [Line2D([0], [0], color='b', label='p(xmiss|xobs) via KDE of MIWAE multiple imp.'),
                    Line2D([0], [0], color='r', label='p(xmiss) via KDE on the fully observed dataset')]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=25)
    plt.xlabel(str(features_i[0]), fontsize=25)
    plt.ylabel(str(features_i[1]), fontsize=25)
    plt.tick_params(axis='both', labelsize = 20)
    plt.title('Subject'+str(i))
    plt.savefig(figures_folder + '/' + file_name +'.pdf',format='pdf',bbox_inches='tight')
    plt.clf()
    plt.close()

