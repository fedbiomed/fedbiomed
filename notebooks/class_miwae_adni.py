import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td

from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager
from fedbiomed.common.constants import ProcessTypes
from typing import Any, Dict, Optional, Union

###########################################################
# Federated standardization                               #
###########################################################

class FedMeanStdTrainingPlan(TorchTrainingPlan):
    
    def init_dependencies(self):
        deps = ["import pandas as pd",
            "import numpy as np"]
        return deps
        
    def init_model(self,model_args):
        
        model = self.MeanStd(model_args)
        
        return model
    
    class MeanStd(nn.Module):
        def __init__(self, model_args):
            super().__init__()
            self.n_features=model_args['n_features']
            
            self.mean = nn.Parameter(torch.zeros(self.n_features,dtype=torch.float64),requires_grad=False)
            self.std = nn.Parameter(torch.zeros(self.n_features,dtype=torch.float64),requires_grad=False)
            self.size = nn.Parameter(torch.zeros(self.n_features,dtype=torch.float64),requires_grad=False)
            self.fake = nn.Parameter(torch.randn(1),requires_grad=True)

        def forward(self, data):
            data_np = data.numpy()
            N = data.shape[0]
            
            ### Implementing with np.nanmean, np.nanstd
            self.size += torch.Tensor([N - np.count_nonzero(np.isnan(data_np[:,dim]))\
                                    for dim in range(self.n_features)])
            self.mean += torch.from_numpy(np.nanmean(data_np,0))
            self.std += torch.from_numpy(np.nanstd(data_np,0))
            
            return self.fake
    
        
    def training_data(self):
        
        df = pd.read_csv(self.dataset_path, sep=',', index_col=False)
        
        ### NOTE: batch_size should be == dataset size ###
        batch_size = df.shape[0]
        x_train = df.values.astype(np.float64)
        #print(x_train.dtype)
        x_mask = np.isfinite(x_train)
        xhat_0 = np.copy(x_train)
        ### NOTE: we keep nan when data is missing
        #xhat_0[np.isnan(x_train)] = 0
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        
        data_manager = DataManager(dataset=xhat_0 , target=x_mask , **train_kwargs)
        
        return data_manager
    
    def training_step(self, data, mask):
        
        return self.model().forward(data)


###########################################################
# Federated miwae                                         #
###########################################################

class MIWAETrainingPlan(TorchTrainingPlan):
    
    def init_dependencies(self):
        deps = ["import torch.distributions as td",
            "import pandas as pd",
            "import numpy as np",
            "from typing import Any, Dict, Optional, Union"]
        return deps
        
    def init_model(self,model_args):
        
        if 'standardization' in model_args:
            self.standardization = True
            self.std_type = model_args['standardization'].get('type')
            if (('fed_mean' in model_args['standardization']) and ('fed_std' in model_args['standardization'])):
                self.fed_mean = np.array(model_args['standardization']['fed_mean']).astype('float32')
                self.fed_std = np.array(model_args['standardization']['fed_std']).astype('float32')
            else:
                self.fed_mean = None
                self.fed_std = None
                
        self.n_features=model_args['n_features']
        self.n_latent=model_args['n_latent']
        self.n_hidden=model_args['n_hidden']
        self.n_samples=model_args['n_samples']
        self.n_samples_test = model_args['n_samples_test'] if 'n_samples_test' in model_args else None
        
        model = self.MIWAE(model_args)
        
        return model
    
    class MIWAE(nn.Module):
        def __init__(self, model_args):
            super().__init__()

            n_features=model_args['n_features']
            n_latent=model_args['n_latent']
            n_hidden=model_args['n_hidden']

            # the encoder will output both the mean and the diagonal covariance
            self.encoder=nn.Sequential(
                            torch.nn.Linear(n_features, n_hidden),
                            #torch.nn.ReLU(),
                            #torch.nn.Linear(n_hidden, n_hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden, 3*n_latent),  
                            )
            # the decoder will output both the mean, the scale, 
            # and the number of degrees of freedoms (hence the 3*p)
            self.decoder = nn.Sequential(
                            torch.nn.Linear(n_latent, n_hidden),
                            #torch.nn.ReLU(),
                            #torch.nn.Linear(n_hidden, n_hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden, 3*n_features),  
                            )

            self.encoder.apply(self.weights_init)
            self.decoder.apply(self.weights_init)

            self.iota = nn.Parameter(torch.zeros(1,n_features),requires_grad=True)

            if model_args['standardization']['type']=='dynamic':
                self.mean = nn.Parameter(torch.zeros(n_features,dtype=torch.float64),requires_grad=False)
                self.std = nn.Parameter(torch.zeros(n_features,dtype=torch.float64),requires_grad=False)
                self.size = nn.Parameter(torch.zeros(n_features,dtype=torch.float64),requires_grad=False)
    
        def weights_init(self,layer):
            if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
    
    def init_optimizer(self,optimizer_args):
        
        #SGD optimizer
        optimizer = torch.optim.SGD(list(self.model().encoder.parameters()) \
                                    + list(self.model().decoder.parameters())\
                                    + [self.model().iota],lr = optimizer_args['lr'])
        
        return optimizer
    
    def miwae_train(self,data,mask,n_samples):
        # prior
        self.p_z = td.Independent(td.Normal(loc=torch.zeros(self.n_latent).to(self._device)\
                                    ,scale=torch.ones(self.n_latent).to(self._device)),1)
        
        batch_size = data.shape[0]
        
        tiledmask = torch.tile(mask,(n_samples,1))
        mask_complement_float = torch.abs(mask-1)

        tilediota = torch.tile(self.model().iota,(data.shape[0],1))
        iotax = data + torch.mul(tilediota,mask_complement_float)
        
        out_encoder = self.model().encoder(iotax)
        
        q_zgivenxobs = td.Independent(td.StudentT(loc=out_encoder[..., :self.n_latent],\
                                                scale=torch.nn.Softplus()(out_encoder[..., self.n_latent:(2*self.n_latent)]),\
                                                df=torch.nn.Softplus()\
                                                (out_encoder[..., (2*self.n_latent):(3*self.n_latent)]) + 3),1)

        zgivenx = q_zgivenxobs.rsample([n_samples])
        zgivenx_flat = zgivenx.reshape([n_samples*batch_size,self.n_latent])

        out_decoder = self.model().decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :self.n_features]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., self.n_features:\
                                                            (2*self.n_features)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()\
        (out_decoder[..., (2*self.n_features):(3*self.n_features)]) + 3

        data_flat = torch.Tensor.repeat(data,[n_samples,1]).reshape([-1,1])

        all_log_pxgivenz_flat = torch.distributions.StudentT\
        (loc=all_means_obs_model.reshape([-1,1]),\
        scale=all_scales_obs_model.reshape([-1,1]),\
        df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([n_samples*batch_size,self.n_features])

        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([n_samples,batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)
        log_imp_weights = logpxobsgivenz + logpz - logq # same importance wieghts used for single and multiple imputation
        params_obs_model = (all_means_obs_model,all_scales_obs_model,all_degfreedom_obs_model)

        return params_obs_model, mask_complement_float, log_imp_weights
    
    def miwae_loss(self,log_imp_weights):
        neg_bound = -torch.mean(torch.logsumexp(log_imp_weights,0))
        return neg_bound

    def training_data(self):
        batch_size=self._training_args.get('batch_size')
        df = pd.read_csv(self.dataset_path, sep=',', index_col=False)
        x_train = df.values.astype(np.float64)
        x_mask = np.isfinite(x_train)
        xhat_0 = np.copy(x_train)
            
        xhat_0[np.isnan(x_train)] = 0
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        
        data_manager = DataManager(dataset=xhat_0 , target=x_mask , **train_kwargs)
        
        return data_manager
    
    def standardize_data(self,data):
        data_norm = np.copy(data)
        if self.std_type == 'static':
            if ((self.fed_mean is not None) and (self.fed_std is not None)):
                print('FEDERATED STANDARDIZATION')
                data_norm = (data_norm - self.fed_mean)/self.fed_std
            else:
                print('LOCAL STANDARDIZATION')
                data_norm = (data_norm - np.nanmean(data_norm,0))/np.nanstd(data_norm,0)
        else:
            if ((torch.count_nonzero(self._model.init_params['mean'])>0) and (torch.count_nonzero(self._model.init_params['std'])>0)):
                print('STANDARDIZATION WITH UPDATED MEAN, STD')
                self.fed_mean = self._model.init_params['mean'].numpy().astype('float32')
                self.fed_std = self._model.init_params['std'].numpy().astype('float32')
            else:
                if ((self.fed_mean is None) and (self.fed_std is None)):
                    print('LOCAL STANDARDIZATION')
                    self.fed_mean = np.nanmean(data_norm,0)
                    self.fed_std = np.nanstd(data_norm,0)
                self._model.init_params['mean'] += torch.from_numpy(self.fed_mean)
                self._model.init_params['std'] += torch.from_numpy(self.fed_std)
            data_norm = (data_norm - self.fed_mean)/self.fed_std
        return data_norm
    
        
    def training_step(self, data, mask):
        self.model().encoder.zero_grad()
        self.model().decoder.zero_grad()
        _,_, log_imp_weights = self.miwae_train(data = data, mask = mask, n_samples=self.n_samples)
        loss = self.miwae_loss(log_imp_weights)
        return loss
    
    def training_routine(self,
                         history_monitor: Any = None,
                         node_args: Union[dict, None] = None,
                         ) -> int:
        
        self._model.init_training()
   
        # Data standardization (only continuous varaibles will be standardized)
        if self.standardization:
            data = self.training_data_loader.dataset.dataset.inputs
            data_std = self.standardize_data(data)
            self.training_data_loader.dataset.dataset.inputs = torch.from_numpy(data_std)

        # training with fed-miwae
        num_samples_observed = super().training_routine(history_monitor, node_args)

        if self.std_type == 'dynamic':
            # impute missing data with current model
            xhat_0 = self.training_data_loader.dataset.dataset.inputs.numpy()
            xhat = np.copy(xhat_0)
            mask = np.copy(self.training_data_loader.dataset.dataset.target.numpy())
            N = xhat.shape[0]

            for i in range(N):
                if (np.sum(mask[i,:])<=self.n_features-1):
                    xhat[i,~mask[i,:].astype(bool)] = self.miwae_impute(
                        data = torch.from_numpy(xhat_0[i,:]).float().reshape([1,self.n_features]),
                        mask = torch.from_numpy(mask[i,:]).float().reshape([1,self.n_features])
                        ).numpy()[0][~mask[i,:].astype(bool)]#.cpu().data.numpy()[0][~mask[i,:]]

            xhat = xhat*self.fed_std + self.fed_mean

            # update local mean, std
            self.model().size += torch.Tensor([N - np.count_nonzero(np.isnan(xhat[:,dim]))\
                                    for dim in range(self.n_features)])
            self.model().mean += torch.from_numpy(np.nanmean(xhat,0))
            self.model().std += torch.from_numpy(np.nanstd(xhat,0))
            
        return num_samples_observed
    
    def miwae_impute(self,data,mask):
        batch_size = data.shape[0]

        with torch.no_grad():
            params_obs_model, mask_complement_float, log_imp_weights = self.miwae_train(data = data, mask = mask, n_samples=self.n_samples_test)

            (all_means_obs_model,all_scales_obs_model,all_degfreedom_obs_model) = params_obs_model
            xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)
            imp_weights = torch.nn.functional.softmax(log_imp_weights,0) # these are w_1,....,w_L for all observations in the batch
            xms = xgivenz.mean.reshape([self.n_samples_test,batch_size,self.n_features])  # that's the only line that changed!
            xm = torch.multiply(torch.einsum('ki,kij->ij', imp_weights, xms),mask_complement_float)

        return xm
    
    # def training_step(self, data, mask):
    #     self.model().encoder.zero_grad()
    #     self.model().decoder.zero_grad()
    #     loss = self.miwae_loss(data = data,mask = mask)
    #     return loss
    
    def testing_step(self, data, mask):
        output = self.model().forward(data)

        #negative log likelihood loss
        loss = torch.nn.functional.nll_loss(output, mask)

        # Returning results as list
        return [loss]
