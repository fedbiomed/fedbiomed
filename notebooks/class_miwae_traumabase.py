import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from sklearn.utils.class_weight import compute_class_weight
from typing import Any, Dict, Optional

from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.training_plans import FedSGDRegressor, FedSGDClassifier
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
            self.n_cov=model_args['n_cov']
            
            self.mean = nn.Parameter(torch.zeros(self.n_features-self.n_cov,dtype=torch.float64),requires_grad=False)
            self.std = nn.Parameter(torch.zeros(self.n_features-self.n_cov,dtype=torch.float64),requires_grad=False)
            self.size = nn.Parameter(torch.zeros(self.n_features-self.n_cov,dtype=torch.float64),requires_grad=False)
            self.fake = nn.Parameter(torch.randn(1),requires_grad=True)

        def forward(self, data):
            data_np = data.numpy()
            N = data.shape[0]
            
            ### Implementing with np.nanmean, np.nanstd
            self.size += torch.Tensor([N - np.count_nonzero(np.isnan(data_np[:,dim]))\
                                    for dim in range(self.n_features-self.n_cov)])
            ###########################################################
            ##Fill nan values with mean
            #col_mean = np.nanmean(data_np,0)
            ##Find indices that you need to replace
            #inds = np.where(np.isnan(data_np))
            ##Place column means in the indices. Align the arrays using take
            #data_np[inds] = np.take(col_mean, inds[1])
            ###########################################################
            self.mean += torch.from_numpy(np.nanmean(data_np,0))
            self.std += torch.from_numpy(np.nanstd(data_np,0))
            
            return self.fake
    
        
    def training_data(self):
        
        df = pd.read_csv(self.dataset_path, sep=',', index_col=False).iloc[:,self.model().n_cov:]
        
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
        self.n_cov=model_args['n_cov']
        self.n_latent=model_args['n_latent']
        self.n_hidden=model_args['n_hidden']
        self.n_samples=model_args['n_samples']
        
        model = self.MIWAE(model_args)
        
        return model
    
    class MIWAE(nn.Module):
        def __init__(self, model_args):
            super().__init__()

            n_features=model_args['n_features']
            n_cov=model_args['n_cov']
            n_variables = n_features-n_cov
            n_latent=model_args['n_latent']
            n_hidden=model_args['n_hidden']

            # the encoder will output both the mean and the diagonal covariance
            self.encoder=nn.Sequential(
                            torch.nn.Linear(n_variables+n_cov, n_hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden, n_hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden, 3*n_latent),  
                            )
            # the decoder will output both the mean, the scale, 
            # and the number of degrees of freedoms (hence the 3*p)
            self.decoder = nn.Sequential(
                            torch.nn.Linear(n_latent+n_cov, n_hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden, n_hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden, 3*n_features),  
                            )

            self.encoder.apply(self.weights_init)
            self.decoder.apply(self.weights_init)

            self.iota = nn.Parameter(torch.zeros(1,n_variables),requires_grad=True)
    
        def weights_init(self,layer):
            if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
    
    def init_optimizer(self,optimizer_args):
        
        #SGD optimizer
        optimizer = torch.optim.SGD(list(self.model().encoder.parameters()) \
                                    + list(self.model().decoder.parameters())\
                                    + [self.model().iota],lr = optimizer_args['lr'])
        
        return optimizer
        
        
    def miwae_loss(self,data,mask,xcat=None):
        # prior
        self.p_z = td.Independent(td.Normal(loc=torch.zeros(self.n_latent).to(self._device)\
                                    ,scale=torch.ones(self.n_latent).to(self._device)),1)
        
        batch_size = data.shape[0]
        n_variables = self.n_features-self.n_cov
        
        tiledmask = torch.tile(mask,(self.n_samples,1))
        mask_complement_float = torch.abs(mask-1)

        tilediota = torch.tile(self.model().iota,(data.shape[0],1))
        iotax = data + torch.mul(tilediota,mask_complement_float)
        
        ############################################################################
        if xcat is None:
            out_encoder = self.model().encoder(iotax)
        else:
            out_encoder = self.model().encoder(torch.cat((iotax,xcat),dim=1))
        ############################################################################
        
        q_zgivenxobs = td.Independent(td.StudentT(loc=out_encoder[..., :self.n_latent],\
                                                scale=torch.nn.Softplus()(out_encoder[..., self.n_latent:(2*self.n_latent)]),\
                                                df=torch.nn.Softplus()\
                                                (out_encoder[..., (2*self.n_latent):(3*self.n_latent)]) + 3),1)

        zgivenx = q_zgivenxobs.rsample([self.n_samples])
        zgivenx_flat = zgivenx.reshape([self.n_samples*batch_size,self.n_latent])

        ############################################################################
        if xcat is None:
            out_decoder = self.model().decoder(zgivenx_flat)
        else:
            out_decoder = self.model().decoder(torch.cat((zgivenx_flat,xcat.repeat(self.n_samples,1)),dim=1))
        ############################################################################
        
        all_means_obs_model = out_decoder[..., :n_variables]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., n_variables:\
                                                            (2*n_variables)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()\
        (out_decoder[..., (2*n_variables):(3*n_variables)]) + 3

        data_flat = torch.Tensor.repeat(data,[self.n_samples,1]).reshape([-1,1])

        all_log_pxgivenz_flat = torch.distributions.StudentT\
        (loc=all_means_obs_model.reshape([-1,1]),\
        scale=all_scales_obs_model.reshape([-1,1]),\
        df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([self.n_samples*batch_size,n_variables])

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
        
        # Data standardization (only continuous varaibles will be standardized)
        if self.standardization:
            x_cov = xhat_0[:,:self.n_cov-1]
            x_cont = xhat_0[:,self.n_cov-1:]
            x_cont = self.standardize_data(x_cont)
            xhat_0 = np.concatenate((x_cov, x_cont), axis=1)
            
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
        loss = self.miwae_loss(data = data[:,self.n_cov:], mask = mask[:,self.n_cov:], xcat = data[:,:self.n_cov])
        #loss = self.miwae_loss(data = data[:,self.n_cov:], mask = mask[:,self.n_cov:], xcat = None)
        return loss

###########################################################
# Federated SGD regressor                                 #
###########################################################

class SGDRegressorTraumabaseTrainingPlan(FedSGDRegressor): 

    def init_dependencies(self):
        deps = ["import torch.distributions as td",
            "import pandas as pd",
            "import numpy as np",
            "from typing import Any, Dict, Optional"]
        return deps

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: Dict[str, Any],
            aggregator_args: Optional[Dict[str, Any]] = None,
        ) -> None:

        self.n_cov=model_args['n_cov']
        self.regressors_col = model_args.get('regressors_col')
        self.target_col = model_args.get('target_col')
        self.n_features = len(self.regressors_col)

        self.standardization = False

        if 'standardization' in model_args:
            self.standardization = True
            if (('fed_mean' in model_args['standardization']) and ('fed_std' in model_args['standardization'])):
                self.fed_mean = np.array(model_args['standardization']['fed_mean'])
                self.fed_std = np.array(model_args['standardization']['fed_std'])
            else:
                self.fed_mean = None
                self.fed_std = None

        super().post_init(model_args, training_args)

    def training_data(self):
        batch_size = self._training_args.get('batch_size')

        dataset = pd.read_csv(self.dataset_path, sep=',', index_col=False)
        
        X = dataset[self.regressors_col].values

        if self.standardization:
            X_cov = X[:,:self.n_cov]
            X_cont = X[:,self.n_cov:]
            X_cont = self.standardize_data(X_cont)
            X = np.concatenate((X_cov, X_cont), axis=1)

        y = dataset[self.target_col]

        return DataManager(dataset=X, target=y.values.astype(int).ravel(), batch_size=batch_size, shuffle=True)
    
    def standardize_data(self,data):
        data_norm = np.copy(data)
        if ((self.fed_mean is not None) and (self.fed_std is not None)):
            print('FEDERATED STANDARDIZATION')
            data_norm = (data_norm - self.fed_mean)/self.fed_std
        else:
            print('LOCAL STANDARDIZATION')
            data_norm = (data_norm - np.nanmean(data_norm,0))/np.nanstd(data_norm,0)
        return data_norm

class FedLogisticRegTraumabase(FedSGDClassifier):
    """Fed-BioMed training plan for scikit-learn Perceptron models.

    This class inherits from FedSGDClassifier, and forces the wrapped
    scikit-learn SGDClassifier model to use a "perceptron" loss, that
    makes it equivalent to an actual scikit-learn Perceptron model.
    """

    _model_dep = (
        "from sklearn.linear_model import SGDClassifier",
        "from fedbiomed.common.training_plans import FedPerceptron"
    )

    def init_dependencies(self):
        deps = ["import torch.distributions as td",
            "import pandas as pd",
            "import numpy as np",
            "from typing import Any, Dict, Optional",
            "from sklearn.linear_model import SGDClassifier",
            "from fedbiomed.common.training_plans import FedSGDClassifier",
            "from sklearn.utils.class_weight import compute_class_weight"]
        return deps

    def __init__(self) -> None:
        """Class constructor."""
        super().__init__()
        self._model.set_params(loss="log")
        #self._model.set_params(penalty="elasticnet")
        #self._model.set_params(class_weight="balanced")

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: Dict[str, Any],
            aggregator_args: Optional[Dict[str, Any]] = None,
        ) -> None:
        model_args["loss"] = "log"#"log_loss"
        #model_args["penalty"] = "elasticnet"
        #model_args["class_weight"]="balanced"
        self.n_cov=model_args['n_cov']
        self.regressors_col = model_args.get('regressors_col')
        self.target_col = model_args.get('target_col')
        self.n_features = len(self.regressors_col)

        self.standardization = False

        if 'standardization' in model_args:
            self.standardization = True
            if (('fed_mean' in model_args['standardization']) and ('fed_std' in model_args['standardization'])):
                self.fed_mean = np.array(model_args['standardization']['fed_mean'])
                self.fed_std = np.array(model_args['standardization']['fed_std'])
            else:
                self.fed_mean = None
                self.fed_std = None

        super().post_init(model_args, training_args)

    def training_data(self):
        batch_size = self._training_args.get('batch_size')

        dataset = pd.read_csv(self.dataset_path, sep=',', index_col=False)
        
        X = dataset[self.regressors_col].values

        if self.standardization:
            X_cov = X[:,:self.n_cov]
            X_cont = X[:,self.n_cov:]
            X_cont = self.standardize_data(X_cont)
            X = np.concatenate((X_cov, X_cont), axis=1)

        y = dataset[self.target_col].values.astype(int).ravel()
        class_weight = compute_class_weight("balanced", np.unique(y), y)
        dict_class_weight = {np.unique(y)[0]:class_weight[0],np.unique(y)[1]:class_weight[1]}
        self._model.set_params(class_weight=dict_class_weight)

        return DataManager(dataset=X, target=y, batch_size=batch_size, shuffle=True)
    
    def standardize_data(self,data):
        data_norm = np.copy(data)
        if ((self.fed_mean is not None) and (self.fed_std is not None)):
            print('FEDERATED STANDARDIZATION')
            data_norm = (data_norm - self.fed_mean)/self.fed_std
        else:
            print('LOCAL STANDARDIZATION')
            data_norm = (data_norm - np.nanmean(data_norm,0))/np.nanstd(data_norm,0)
        return data_norm
