import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td

from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager
from fedbiomed.common.constants import ProcessTypes

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
            "import numpy as np"]
        return deps
        
    def init_model(self,model_args):
        
        if 'standardization' in model_args:
            self.standardization = True
            if (('fed_mean' in model_args['standardization']) and ('fed_std' in model_args['standardization'])):
                self.fed_mean = np.array(model_args['standardization']['fed_mean'])
                self.fed_std = np.array(model_args['standardization']['fed_std'])
            else:
                self.fed_mean = None
                self.fed_std = None
                
        self.n_features=model_args['n_features']
        self.n_latent=model_args['n_latent']
        self.n_hidden=model_args['n_hidden']
        self.n_samples=model_args['n_samples']
        
        model = self.MIWAE(model_args)
        
        return model
    
    class MIWAE(nn.Module):
        def __init__(self, model_args):
            super().__init__()

            n_features=model_args['n_features']
            n_latent=model_args['n_latent']
            n_hidden=model_args['n_hidden']
            n_samples=model_args['n_samples']

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
    
        def weights_init(self,layer):
            if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
    
    def init_optimizer(self,optimizer_args):
        
        #SGD optimizer
        optimizer = torch.optim.SGD(list(self.model().encoder.parameters()) \
                                    + list(self.model().decoder.parameters())\
                                    + [self.model().iota],lr = optimizer_args['lr'])
        
        return optimizer
        
        
    def miwae_loss(self,data,mask):
        # prior
        self.p_z = td.Independent(td.Normal(loc=torch.zeros(self.n_latent).to(self._device)\
                                    ,scale=torch.ones(self.n_latent).to(self._device)),1)
        
        batch_size = data.shape[0]
        
        tiledmask = torch.tile(mask,(self.n_samples,1))
        mask_complement_float = torch.abs(mask-1)

        tilediota = torch.tile(self.model().iota,(data.shape[0],1))
        iotax = data + torch.mul(tilediota,mask_complement_float)
        
        out_encoder = self.model().encoder(iotax)
        #out_encoder = self.model().encoder(iota_x)
        
        #q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :self.n_latent],\
        #                                        scale=torch.nn.Softplus()\
        #                                        (out_encoder[..., self.n_latent:\
        #                                                    (2*self.n_latent)])),1)
        q_zgivenxobs = td.Independent(td.StudentT(loc=out_encoder[..., :self.n_latent],\
                                                scale=torch.nn.Softplus()(out_encoder[..., self.n_latent:(2*self.n_latent)]),\
                                                df=torch.nn.Softplus()\
                                                (out_encoder[..., (2*self.n_latent):(3*self.n_latent)]) + 3),1)

        zgivenx = q_zgivenxobs.rsample([self.n_samples])
        zgivenx_flat = zgivenx.reshape([self.n_samples*batch_size,self.n_latent])

        out_decoder = self.model().decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :self.n_features]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., self.n_features:\
                                                            (2*self.n_features)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()\
        (out_decoder[..., (2*self.n_features):(3*self.n_features)]) + 3

        data_flat = torch.Tensor.repeat(data,[self.n_samples,1]).reshape([-1,1])
        #tiledmask = torch.Tensor.repeat(mask,[self.n_samples,1])

        all_log_pxgivenz_flat = torch.distributions.StudentT\
        (loc=all_means_obs_model.reshape([-1,1]),\
        scale=all_scales_obs_model.reshape([-1,1]),\
        df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([self.n_samples*batch_size,self.n_features])

        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([self.n_samples,batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))

        return neg_bound

    def training_data(self):
        batch_size=self._training_args.get('batch_size')
        df = pd.read_csv(self.dataset_path, sep=',', index_col=False)
        x_train = df.values.astype(np.float64)
        x_mask = np.isfinite(x_train)
        # xhat_0: missing values are replaced by zeros. 
        #This x_hat0 is what will be fed to our encoder.
        xhat_0 = np.copy(x_train)
        
        # Data standardization
        if self.standardization:
            xhat_0 = self.standardize_data(xhat_0)
            
        xhat_0[np.isnan(x_train)] = 0
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        
        data_manager = DataManager(dataset=xhat_0 , target=x_mask , **train_kwargs)
        
        return data_manager
    
    def standardize_data(self,data):
        data_norm = np.copy(data)
        if ((self.fed_mean is not None) and (self.fed_std is not None)):
            print('FEDERATED STANDARDIZATION')
            data_norm = (data_norm - self.fed_mean)/self.fed_std
        else:
            print('LOCAL STANDARDIZATION')
            data_norm = (data_norm - np.nanmean(data_norm,0))/np.nanstd(data_norm,0)
        return data_norm
    
    def training_step(self, data, mask):
        self.model().encoder.zero_grad()
        self.model().decoder.zero_grad()
        loss = self.miwae_loss(data = data,mask = mask)
        return loss
    
    def testing_step(self, data, mask):
        output = self.model().forward(data)

        #negative log likelihood loss
        loss = torch.nn.functional.nll_loss(output, mask)

        # Returning results as list
        return [loss]
