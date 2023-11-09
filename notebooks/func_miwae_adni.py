import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import csv, os
import torch.distributions as td
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns; sns.set()

###########################################################
#Define the imputation and the MSE functions              #
###########################################################

def encoder_decoder_iota_opt(p,h,d,lr):
    encoder = nn.Sequential(
        torch.nn.Linear(p, h),
        #torch.nn.ReLU(),
        #torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, 3*d),  
    )

    decoder = nn.Sequential(
        torch.nn.Linear(d, h),
        #torch.nn.ReLU(),
        #torch.nn.Linear(h, h),
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

def miwae_impute_single(encoder,decoder,iota,data,mask,p,d,L):

    p_z = td.Independent(td.Normal(loc=torch.zeros(d),scale=torch.ones(d)),1)

    batch_size = data.shape[0]
    
    tiledmask = torch.tile(mask,(L,1))
    mask_complement_float = torch.abs(mask-1)

    tilediota = torch.tile(iota,(data.shape[0],1))
    iotax = data + torch.mul(tilediota,mask_complement_float)
    
    out_encoder = encoder(iotax)
    #q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
    q_zgivenxobs = td.Independent(td.StudentT(loc=out_encoder[..., :d],\
                                                scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)]),\
                                                df=torch.nn.Softplus()\
                                                (out_encoder[..., (2*d):(3*d)]) + 3),1)

    zgivenx = q_zgivenxobs.rsample([L])
    zgivenx_flat = zgivenx.reshape([L*batch_size,d])

    out_decoder = decoder(zgivenx_flat)
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

def miwae_impute(encoder,decoder,iota, data, mask,d,p,L,kind="single",num_samples=20):
    with torch.no_grad():

        # kind = "single" will return only the single imputation
        # kind = "multiple" will return num_samples multiple imputations
        # kind = "both" will both the single imputation and num_samples multiple imputations

        p_z = td.Independent(td.Normal(loc=torch.zeros(d),scale=torch.ones(d)),1)

        batch_size = data.shape[0]
        
        tiledmask = torch.tile(mask,(L,1))
        mask_complement_float = torch.abs(mask-1)

        tilediota = torch.tile(iota,(data.shape[0],1))
        iotax = data + torch.mul(tilediota,mask_complement_float)
        
        out_encoder = encoder(iotax)
        #q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
        q_zgivenxobs = td.Independent(td.StudentT(loc=out_encoder[..., :d],\
                                                    scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)]),\
                                                    df=torch.nn.Softplus()\
                                                    (out_encoder[..., (2*d):(3*d)]) + 3),1)

        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([L*batch_size,d])

        out_decoder = decoder(zgivenx_flat)
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
            #xmul = torch.reshape(xhat_0_batch + torch.multiply(torch.reshape(xgivenz.sample(),[L,xhat_0_batch.shape[0],p]),mask_complement_float),[L,xhat_0_batch.shape[0],p])
            xmul = torch.reshape(torch.multiply(torch.reshape(xgivenz.sample(),[L,batch_size,p]),mask_complement_float),[L,batch_size,p])
            xmul_gat = xmul.permute(1,0,2)[:,torch.t(sir)][0]
            # tf.gather(tf.transpose(xmul,perm=[1,0,2]), tf.transpose(sir), axis = 1, batch_dims=1)

    if (kind=="single"):
        return xm

    else:
        return xm, xmul_gat

# def mse(xhat,xtrue,mask): # MSE function for imputations
#     xhat = np.array(xhat)
#     xtrue = np.array(xtrue)
#     return np.mean(np.power(xhat-xtrue,2)[~mask])

def mse(xhat,xtrue,mask=None,normalized=False): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    if mask is not None:
        MSE = np.mean(np.power(xhat-xtrue,2)[~mask])
        NMSE = MSE/np.mean(np.power(xtrue,2)[~mask])
    else:
        MSE = np.mean(np.power(xhat-xtrue,2))
        NMSE = MSE/np.mean(np.power(xtrue,2))
    return NMSE if normalized else MSE

def miwae_loss(encoder, decoder, iota, data, mask, d, p, K):
    p_z = td.Independent(td.Normal(loc=torch.zeros(d),scale=torch.ones(d)),1)
    batch_size = data.shape[0]
        
    tiledmask = torch.tile(mask,(K,1))
    mask_complement_float = torch.abs(mask-1)

    tilediota = torch.tile(iota,(data.shape[0],1))
    iotax = data + torch.mul(tilediota,mask_complement_float)
    
    out_encoder = encoder(iotax)
    #out_encoder = encoder(iota_x)
    
    #q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
    q_zgivenxobs = td.Independent(td.StudentT(loc=out_encoder[..., :d],\
                                            scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)]),\
                                            df=torch.nn.Softplus()\
                                            (out_encoder[..., (2*d):(3*d)]) + 3),1)

    zgivenx = q_zgivenxobs.rsample([K])
    zgivenx_flat = zgivenx.reshape([K*batch_size,d])

    out_decoder = decoder(zgivenx_flat)
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

def recover_data(data_missing, data_full, fed_mean = None, fed_std = None):

    # TEST DATA: data_missing = data_test_missing; data_full = data_test
    # CLIENT i, i=0,1,2: data_missing = Clients_missing[i]; data_full = Clients_data[i]

    xmiss = np.copy(data_missing)
    mask = np.isfinite(xmiss) # binary mask that indicates which values are missing

    # Evaluate local mean and std of test dataset
    mean = np.nanmean(xmiss,0)
    std = np.nanstd(xmiss,0)

    # local standardization
    xmiss_local_std = np.copy(data_missing)
    xmiss_local_std = (xmiss_local_std - mean)/std
    xhat_0_local_std = np.copy(xmiss_local_std)
    xhat_0_local_std[np.isnan(xmiss_local_std)] = 0
    xhat_local_std = np.copy(xhat_0_local_std) # This will be out imputed data matrix
    xfull_local_std = np.copy(data_full)
    xfull_local_std = (xfull_local_std - mean)/std

    if ((fed_mean is not None) and (fed_std is not None)): 
        if ((type(fed_mean) != np.ndarray) and (type(fed_std) != np.ndarray)):
            fed_mean, fed_std = fed_mean.numpy(), fed_std.numpy()
        # standardization with respect to the federated dataset
        xmiss_global_std = np.copy(data_missing)
        xmiss_global_std = (xmiss_global_std - fed_mean)/fed_std
        xhat_0_global_std = np.copy(xmiss_global_std)
        xhat_0_global_std[np.isnan(xmiss_global_std)] = 0
        xhat_global_std = np.copy(xhat_0_global_std) # This will be out imputed data matrix
        xfull_global_std = np.copy(data_full)
        xfull_global_std = (xfull_global_std - fed_mean)/fed_std

        return xmiss, mask, xhat_global_std, xfull_global_std, xhat_local_std, xfull_local_std, mean, std
    else:
        return xmiss, mask, xhat_local_std, xfull_local_std, mean, std


def testing_func(features,data_missing, data_full, mask, encoder, decoder, iota, d,L,
                 idx_cl=4,result_folder='results',method='FedAvg',kind="single",num_samples=20,
                 mean = None, std = None, do_figures = False, stat_mul = None):

    xhat = np.copy(data_missing)
    xhat_0 = np.copy(data_missing)
    xfull = np.copy(data_full)

    p = data_full.shape[1] # number of features
    n = data_full.shape[0] # number of features

    if ((mean is not None) and (std is not None)):
        if ((type(mean) != np.ndarray) and (type(std) != np.ndarray)):
            mean, std = mean.numpy(), std.numpy()
        normalized = True

    if kind == 'single':
        xhat_single = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, data = torch.from_numpy(xhat_0).float(),
                               mask = torch.from_numpy(mask).float(),p=p, d = d,L= L)        
        xhat[~mask] = xhat_single.cpu().data.numpy()[~mask]

    else:
        xhat_mul = np.copy(xhat_0)
        xfull_mul = np.copy(xfull)
        Err_mul_mean = [np.copy(xhat_0) for  _ in range(num_samples + 1)]
        for i in range(n):
            if (np.sum(mask[i,:])<=p-1):
                xhat_single, xhat_multiple = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, 
                                                        data = torch.from_numpy(xhat_0[i,:]).reshape([1,p]).float(), 
                                                        mask = torch.from_numpy(mask[i,:]).reshape([1,p]).float(),
                                                        p=p, d = d,L= L,kind=kind,num_samples=num_samples)

                true_values = xfull[i,~mask[i,:].astype(bool)]
                single_imp = np.squeeze(xhat_single[:,~mask[i,:].astype(bool)])
                xhat[i,~mask[i,:].astype(bool)] = single_imp.numpy()
                mul_imp = np.squeeze(xhat_multiple.numpy()[:,:,~mask[i,:].astype(bool)])
                features_i = np.array(features)[~mask[i,:].astype(bool)]
                xfull_mask = np.copy(xfull)[:,~mask[i,:].astype(bool)]
                if ((do_figures) and (np.sum(mask[i,:])<=p-2)):
                    multiple_imputation_plot(result_folder,xfull_mask,mul_imp,single_imp,true_values,method,
                                            idx_cl,i,features_i,std,mean,mask[i,:],normalized)
                mul_imp_i = np.append(mul_imp,[single_imp.numpy()],axis=0)

                if stat_mul == 'MSE_of_mean':
                    mul_imp_mean_i = np.mean(mul_imp_i,axis=0)
                    xhat_mul[i,~mask[i,:].astype(bool)] = mul_imp_mean_i

                else:
                    for nsam in range(num_samples+1):
                        Err_mul_mean[nsam][i,~mask[i,:].astype(bool)]=mul_imp_i[nsam]#mul_imp_i[nsam,:]

    err_standardized = float(np.array([mse(xhat,xfull,mask)]))

    if normalized:
        xhat_destd = np.copy(xhat)
        xhat_destd = xhat_destd*std + mean
        xfull_destd = np.copy(xfull)
        xfull_destd = xfull_destd*std + mean
        err = float(np.array([mse(xhat_destd,xfull_destd,mask,normalized=normalized)]))
        if kind != 'single':
            xfull_mul = xfull_mul*std + mean 
            if stat_mul == 'MSE_of_mean':
                xhat_mul = xhat_mul*std + mean
            else:
                for nsam in range(num_samples+1):
                    Err_mul_mean[nsam] = Err_mul_mean[nsam]*std + mean
    if kind != 'single':                        
        if stat_mul == 'MSE_of_mean':
            err_mul = float(np.array([mse(xhat_mul,xfull_mul,mask,normalized=normalized)]))
        else:
            Errs_mul = [float(np.array([mse(Err_mul_mean[i],xfull_mul,mask,normalized=normalized)])) for i in range(num_samples+1)]
            err_mul=float(np.mean(Errs_mul))
            min_mul = float(np.min(Errs_mul))
            max_mul = float(np.max(Errs_mul))
            std_mul = float(np.std(Errs_mul))

    if (kind=="single"):
        return [err_standardized] if normalized==False else [err_standardized,err]
    else:
        if normalized==False:
            return [err_standardized,err_mul]  if stat_mul == 'MSE_of_mean' else [err_standardized,err_mul,min_mul,max_mul,std_mul]
        else:
            return [err_standardized,err,err_mul] if stat_mul == 'MSE_of_mean' else [err_standardized,err,err_mul,min_mul,max_mul,std_mul]

        

def testing_func_deprecated(features,data_missing, data_full, mask, encoder, decoder, iota, d,L,
                 idx_cl=4,result_folder='results',method='FedAvg',kind="single",num_samples=20,
                 mean = None, std = None, do_figures = False, stat_mul = None):

#'mean_RMSE','mean_MSE','MSE_of_mean'

    #features = data_full.columns.values.tolist()
    xhat = np.copy(data_missing)
    xhat_0 = np.copy(data_missing)
    xfull = np.copy(data_full)

    p = data_full.shape[1] # number of features

    xm = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, data = torch.from_numpy(xhat_0).float(),
                               mask = torch.from_numpy(mask).float(),p=p, d = d,L= L)
        
    xhat[~mask] = xm.cpu().data.numpy()[~mask]

    if ((mean is not None) and (std is not None)):
        if ((type(mean) != np.ndarray) and (type(std) != np.ndarray)):
            mean, std = mean.numpy(), std.numpy()
        xhat_destd = np.copy(xhat)
        xhat_destd = xhat_destd*std + mean
        xfull_destd = np.copy(xfull)
        xfull_destd = xfull_destd*std + mean
        err_standardized = float(np.array([mse(xhat,xfull,mask)]))
        err = float(np.array([mse(xhat_destd,xfull_destd,mask,normalized=True)]))
        normalized = True
    else:
        normalized = False
        err = float(np.array([mse(xhat,xfull,mask)]))

    if (kind=="single"):
        return [err] if normalized==False else [err_standardized,err]
    else:
        n = data_full.shape[0]
        xhat_mul = np.copy(xhat_0)
        #xhat_mul_median = np.copy(xhat_0)
        xfull_mul = np.copy(xfull) if normalized==False else xfull_destd
        Err_mul_mean = [np.copy(xhat_0) for  _ in range(num_samples + 1)]
        for i in range(n):
            if (np.sum(mask[i,:])<=p-2):
                xhat_single, xhat_multiple = miwae_impute(encoder = encoder, decoder = decoder, iota = iota, 
                                                        data = torch.from_numpy(xhat_0[i,:]).reshape([1,p]).float(), 
                                                        mask = torch.from_numpy(mask[i,:]).reshape([1,p]).float(),
                                                        p=p, d = d,L= L,kind=kind,num_samples=num_samples)


                xfull_mask = np.copy(xfull)[:,~mask[i,:].astype(bool)]
                true_values = xfull[i,~mask[i,:].astype(bool)]
                single_imp = np.squeeze(xhat_single[:,~mask[i,:].astype(bool)])
                mul_imp = np.squeeze(xhat_multiple.numpy()[:,:,~mask[i,:].astype(bool)])
                features_i = np.array(features)[~mask[i,:].astype(bool)]
                if do_figures:
                    multiple_imputation_plot(result_folder,xfull_mask,mul_imp,single_imp,true_values,method,idx_cl,i,features_i,std,mean,mask[i,:].astype(bool),normalized)
                mul_imp_i = np.append(mul_imp,[single_imp.numpy()],axis=0)

                if stat_mul == 'MSE_of_mean':
                    mul_imp_mean_i = np.mean(mul_imp_i,axis=0)
                    #mul_imp_median_i = np.median(mul_imp_i,axis=0)
                    xhat_mul[i,~mask[i,:].astype(bool)] = mul_imp_mean_i
                    #xhat_mul_median[i,~mask[i,:].astype(bool)] = mul_imp_median_i 

                else:
                    for nsam in range(num_samples+1):
                        Err_mul_mean[nsam][i,~mask[i,:].astype(bool)]=mul_imp_i[nsam,:]
                        #mul_imp_i_ns = np.copy(xhat_0[i,:])
                        #mul_imp_i_ns[~mask[i,:].astype(bool)] = mul_imp_i[nsam,:]
                        #if normalized:
                        #    mul_imp_i_ns = mul_imp_i_ns*std[~mask[i,:].astype(bool)] + mean[~mask[i,:].astype(bool)]
                        ##Err_mul_mean[nsam].append(float(np.array([mse(mul_imp_i_ns,xfull_mul[i,~mask[i,:].astype(bool)],mask[i,~mask[i,:].astype(bool)],normalized=normalized)])))
                        #Err_mul_mean[nsam].append(np.power(mul_imp_i_ns-xfull_mul[i,:],2))
                        
        if stat_mul == 'MSE_of_mean':
            if normalized:
                xhat_mul = xhat_mul*std + mean
                #xhat_mul_median = xhat_mul_median*std + mean
            err_mul = float(np.array([mse(xhat_mul,xfull_mul,mask,normalized=normalized)]))
            #err_mul_median = float(np.array([mse(xhat_mul_median,xfull_mul,mask,normalized=normalized)]))
        else:
            if normalized:
                for nsam in range(num_samples+1):
                    Err_mul_mean[nsam] = Err_mul_mean[nsam]*std + mean
            ##err_mul=float(np.mean([np.mean(Err_mul_mean[i]) for i in range(num_samples+1)]))
            #Errs_mul = [np.mean(Err_mul_mean[i]) for i in range(num_samples+1)] if normalized == False else \
            #        [np.mean(Err_mul_mean[i])/np.mean(np.power(xfull_mul,2)[~mask]) for i in range(num_samples+1)]
            Errs_mul = [float(np.array([mse(Err_mul_mean[i],xfull_mul,mask,normalized=normalized)])) for i in range(num_samples+1)]
            err_mul=float(np.mean(Errs_mul))
            min_mul = float(np.min(Errs_mul))
            max_mul = float(np.max(Errs_mul))
            std_mul = float(np.std(Errs_mul))

        
        #print(err_mul,err_mul_median,err)
        if normalized==False:
            return [err,err_mul] 
        else:
            return [err_standardized,err,err_mul] if stat_mul == 'MSE_of_mean' else [err_standardized,err,err_mul,min_mul,max_mul,std_mul]

def save_results(result_folder, Split_type,Train_data,Test_data,
                perc_missing_train,perc_missing_test,model,
                N_train_centers,Size,N_rounds,N_epochs,
                std_training,std_testing,MSE):

    os.makedirs(result_folder, exist_ok=True) 
    exp_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output_file_name = 'output_'+str(exp_id)+'_'+str(np.random.randint(9999, dtype=int))+'.csv'
    fieldnames=['Split_type', 'Train_data', 'Test_data', 'perc_missing_train', 
                'perc_missing_test', 'model', 
                'N_train_centers', 'Size', 'N_rounds', 'N_epochs',
                'std_training', 'std_testing', 'MSE']
    if not os.path.exists(result_folder+'/'+output_file_name):
        output = open(result_folder+'/'+output_file_name, "w")
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter = ';')
        writer.writeheader()
        output.close()

    # Dictionary to be added
    dict_out={'Split_type': Split_type, 'Train_data': Train_data, 'Test_data': Test_data, 
                'perc_missing_train': perc_missing_train, 'perc_missing_test': perc_missing_test, 
                'model': model, 'N_train_centers': N_train_centers, 'Size': Size, 
                'N_rounds': N_rounds, 'N_epochs': N_epochs,
                'std_training': std_training, 'std_testing': std_testing, 'MSE': MSE}

    with open(result_folder+'/'+output_file_name, 'a') as output_file:
        dictwriter_object = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter = ';')
        dictwriter_object.writerow(dict_out)
        output_file.close()

def databases(data_folder,Split_type,idx_clients,idx_Test_data,N_cl,root_dir=None):

    if Split_type == 'notiid':
        data_folder += 'ADNI_notiid'
    elif Split_type == 'site_1':
        data_folder += 'ADNI_site_1'
    elif Split_type == 'site_2':
        data_folder += 'ADNI_site_2'

    if root_dir is not None:
        root_dir = Path.home() if root_dir == 'home' else Path.home().joinpath( 'Documents/INRIA_EPIONE/FedBioMed', 'fedbiomed' )
        data_folder = root_dir.joinpath(data_folder)
     
    if Split_type == 'site_2':
        Perc_missing = [0.3,0.2,0.4,0.1]
        Perc_missing_test = Perc_missing[idx_Test_data-1]
        Perc_missing.remove(Perc_missing_test)
    else:
        Perc_missing = [0.3 for _ in range(N_cl)]
        Perc_missing_test = 0.3

    Clients_data=[]
    Clients_missing=[]
    for i in idx_clients:
        data_full_file = os.path.join(str(data_folder), "dataset_full_"+str(i)+".csv")
        #data_full_file = data_folder.joinpath("dataset_full_"+str(i)+".csv")
        data_full = pd.read_csv(data_full_file, sep=",",index_col=False)
        Clients_data.append(data_full)
        data_file = os.path.join(str(data_folder),"dataset_"+str(i)+".csv")
        #data_file = data_folder.joinpath("dataset_"+str(i)+".csv")
        data = pd.read_csv(data_file, sep=",",index_col=False)
        Clients_missing.append(data)

    test_file = os.path.join(str(data_folder),"dataset_full_"+str(idx_Test_data)+".csv")
    #test_file = data_folder.joinpath("dataset_full_"+str(idx_Test_data)+".csv")
    data_test = pd.read_csv(test_file, sep=",",index_col=False)
    test_missing_file = os.path.join(str(data_folder),"dataset_"+str(idx_Test_data)+".csv")
    #test_missing_file = data_folder.joinpath("dataset_"+str(idx_Test_data)+".csv")
    data_test_missing = pd.read_csv(test_missing_file, sep=",",index_col=False)

    return Clients_data, Clients_missing, data_test, data_test_missing, Perc_missing, Perc_missing_test

def multiple_imputation_plot(result_folder,xfull,mul_imp,single_imp,true_values,method,idx_cl,i,features_i,std,mean,mask_i,normalized):
    figures_folder = result_folder+'/Figures'
    os.makedirs(figures_folder, exist_ok=True)
    exp_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'_'+str(np.random.randint(9999, dtype=int))
    file_name = 'Multiple_imp_'+method+exp_id+'Client_'+str(idx_cl)
    ax = plt.figure(figsize=(9,10))
    if normalized == True:
        xfull = xfull*std[~mask_i.astype(bool)]+mean[~mask_i.astype(bool)]
        mul_imp = mul_imp*std[~mask_i.astype(bool)]+mean[~mask_i.astype(bool)]
        single_imp = single_imp*std[~mask_i.astype(bool)]+mean[~mask_i.astype(bool)]
        true_values = true_values*std[~mask_i.astype(bool)]+mean[~mask_i.astype(bool)]
    feature_idx = np.random.choice(range(len(single_imp)),2,replace=False)
    i0, i1 = int(feature_idx[0]), int(feature_idx[1])
    sns.kdeplot(x=mul_imp[:,i0],y=mul_imp[:,i1],fill=True)
    plt.scatter(x=single_imp[i0],y=single_imp[i1], s =100, marker = "P",  c = '#2ca02c')
    plt.scatter(x=true_values[i0],y=true_values[i1], s =100, marker = "X",  c = '#d62728')
    legend_elements = [Line2D([0], [0], color='b', lw=4, label='MIWAE (KDE of multiple imp.)'),
                    Line2D([0], [0], marker='P', color='w', markerfacecolor = '#2ca02c', label='MIWAE (single imp.)', markersize=20, lw=0),
                    Line2D([0], [0], marker='X', color='w', markerfacecolor = '#d62728', label='True value', markersize=20, lw=0)]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=25)
    plt.xlabel(str(features_i[i0]), fontsize=25)
    plt.ylabel(str(features_i[i1]), fontsize=25)
    plt.tick_params(axis='both', labelsize = 20)
    plt.title('Subject'+str(i))
    plt.savefig(figures_folder + '/' + file_name +'.pdf',format='pdf',bbox_inches='tight')
    plt.clf()
    plt.close()

    ax = plt.figure(figsize=(9,10))
    file_name = 'px_miss_px_obs_'+method+exp_id+'Client_'+str(idx_cl)
    sns.kdeplot(x=mul_imp[:,i0],y=mul_imp[:,i1], color = 'b')
    sns.kdeplot(x=xfull[:,i0],y=xfull[:,i1], color = 'r')
    legend_elements = [Line2D([0], [0], color='b', label='p(xmiss|xobs) via KDE of MIWAE multiple imp.'),
                    Line2D([0], [0], color='r', label='p(xmiss) via KDE on the fully observed dataset')]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=25)
    plt.xlabel(str(features_i[i0]), fontsize=25)
    plt.ylabel(str(features_i[i1]), fontsize=25)
    plt.tick_params(axis='both', labelsize = 20)
    plt.title('Subject'+str(i))
    plt.savefig(figures_folder + '/' + file_name +'.pdf',format='pdf',bbox_inches='tight')
    plt.clf()
    plt.close()

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