from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import preprocessing
from copy import deepcopy
from scipy.stats import matrix_normal, invgamma
from numpy.linalg import solve
from math import log
from fedbiomed.common.logger import logger
from fedbiomed.common.pythonmodel import PythonModelPlan
from fedbiomed.common.multiview_parser import MultiViewCSVParser


class PpcaPlan(PythonModelPlan):
    def __init__(self,
                 kwargs):
        super(PpcaPlan, self).__init__(kwargs)
        """
           Class initializer.
           :kwargs (dictionary) containing the total number of observed views (tot_views),
                                           the dimension of each view (dim_views), 
                                           the dimension of the latent space (n_components)
                                           a boolean to decide if data have to be normalized (is_norm)
                                           priors for one or more local parameters (not mandatory)
        """
        # list dependencies of the model
        self.dependencies = [
                             "from fedbiomed.common.ppca import PpcaPlan",
                             "import pandas as pd"
                             ]

        self.dataset_path = None
                        
        self.K = kwargs['tot_views']
        self.views_names = kwargs.get('views_names', range(self.K))
        self.dim_views = kwargs['dim_views']
        self.n_components = kwargs['n_components']
        self.is_norm = kwargs['is_norm']

        self.params_dict = {'K': self.K,
                            'dimensions': (self.dim_views,self.n_components),
                            'views_names': self.views_names,
                            # local params:
                            'Wk': None, 
                            'muk': None,
                            'sigma2k': None,
                            # global params:
                            'tilde_muk': {key: None for key in self.views_names},
                            'tilde_Wk': {key: None for key in self.views_names},
                            'tilde_Sigma2k': {key: None for key in self.views_names},
                            'Alpha': {key: None for key in self.views_names},
                            'Beta': {key: None for key in self.views_names},
                            'sigma_til_muk': {key: None for key in self.views_names},
                            'sigma_til_Wk': {key: None for key in self.views_names},
                            'sigma_til_sigma2k': {key: None for key in self.views_names}}

        # if priors are given as model args, corresponding global parameter are uploaded
        # and initialization of local parameter are done using prior from the first round,
        # and MAP optimization is performed instead of EM.
        # The researcher can have prior information only over some parameters
        if (('tilde_muk' in kwargs) and ('sigma_til_muk' in kwargs)):
            self.update_params({'tilde_muk': kwargs['tilde_muk'], 'sigma_til_muk': kwargs['sigma_til_muk']})
        
        if (('tilde_Wk' in kwargs) and ('sigma_til_Wk' in kwargs)):
            self.update_params({'tilde_Wk': kwargs['tilde_Wk'], 'sigma_til_Wk': kwargs['sigma_til_Wk']})
        if (('sigma_til_sigma2k' in kwargs) and ('Alpha' in kwargs) and ('Beta' in kwargs)):
            self.update_params({'sigma_til_sigma2k': kwargs['sigma_til_sigma2k'], 'Alpha': kwargs['Beta'], 'Beta': kwargs['Beta']})
        
        self.views_iterator = range(self.K) if self.K is not None else None
        

    #################################################
    def training_routine(self, 
                         n_iterations: int,
                         log_interval: int = 3,
                         monitor=None):

        """ 
        Args:
            n_iterations (int): the number of EM/MAP iterations for the current round.
            log_interval (int): Interval for iteration to send logs/scalar values 
        """

        #use_cuda = torch.cuda.is_available()
        #device = torch.device("cuda" if use_cuda else "cpu")
        #self.device = "cpu"
        # labels can be provided or not. Note that to perform optimization, 
        # only data and information on observed views should be provided.
        
        
        # if len(self.training_data()) == 4:
        #     X, Xk, ViewsX, y = self.training_data()
        # elif len(self.training_data()) == 3:
        #     X, Xk, ViewsX = self.training_data()
        # else: 
        #     raise ValueError(f"unexpected number of value to unpack (expecting 3 or 4, got {len(self.training_data)}")
        dataset = self.training_data()
        X, Xk, ViewsX, _ = self.parse_input_values(dataset)
        N = X.shape[0]  # nb of samples in dataset
        q = self.n_components  # latent dim
        D_i = self.dim_views  # nb of features per views
        
        # self.q_i contains the "effective" latent space dimension per view 
        # (at least equal to original view dimension-1)

        q_i = [] 
        for i in D_i:
            if i <= q:
                q_i.append(i - 1)
            else:
                q_i.append(q)

        # initialize W and Sigma2 either randomly or using priors if available
        Wk, Sigma2 = self.initial_loc_params(q_i,ViewsX)

        # # epochs indices for saving results
        # sp_arr = np.arange(1, n_iterations + 1)[np.round(
        #     np.linspace(0, len(np.arange(1, n_iterations + 1)) - 1, int(n_iterations / 3))).astype(
        #     int)]

        # training loop
        
        for i in range(n_iterations):
            muk, Wk, Sigma2, ELL = self.EM_Optimization(N,q_i,Xk,Wk,Sigma2,ViewsX)
            
            if i % log_interval == 0:
                logger.info('Iteration: [{}/{}]\tExpected LL: {:.6f}'.format(
                    i,
                    n_iterations,
                    ELL))

                # Send scalar values (ELL) via general/monitoring topic
                if monitor is not None:
                    # -1 means no mini-batch (traning with samples samples)
                    monitor.add_scalar("Log Likelihood", ELL, -1 , i)

        # update local parameters
        #self.load_params({'Wk': Wk, 'muk': muk, 'sigma2k': Sigma2})
        self.update_params({'Wk': Wk, 'muk': muk, 'sigma2k': Sigma2})

    def parse_input_values(self,
                           dataset: pd.DataFrame,
                           target: pd.Series = None) -> Tuple[pd.DataFrame,
                                                              List[pd.DataFrame],
                                                              List[int]]:
                               
        multiview_df = MultiViewCSVParser(dataset,
                                          self.is_multi_view,
                                          self.dim_views)
        Xk = []  # a list containing dataframes (if availaible)
        # or np.nan (if not available)
        ViewsX = []  # specify if views is present in node or not
        ind = 0
        self.views_iterator = self.views_names
        # if self.is_multi_view:
        #     # get a list of all view names
        #     iterator = sorted(set(dataset.columns.get_level_values(0)))
        #     iterator = list(iterator)
            
        #     def pandas_handler(df, x):
        #         return df[x]  
        # else:
        #     iterator = range(self.K)
        #     def pandas_handler(df, x):
        #         return df.iloc[:, ind:ind + self.dim_views[x]]
        # self.dim_views = multiview_df.views
        # iterate over number of views
        # the followig is for parsing input values, 
        # dealing with missing data (should be NaN datasets),
        # normalize datasets  and creating Xk and ViewsX
        for k, iter_elem in enumerate(self.views_iterator):
            if multiview_df[iter_elem].isnull().values.any():
                # case where a view is missing 
                # (it should contain an array of nan)
                
                Xk.append(np.nan)
                #Xk.append('NaN')
                ViewsX.append(0)
            else:
                # if norm = true, data are normalized with min max scaler
                X_k = multiview_df[iter_elem]
                if self.is_norm:
                    X_k = self.normalize_data(X_k) 
                    
                    
                Xk.append(X_k)
                
                ViewsX.append(1)
            ind += self.dim_views[k]  # only used for non multiview datasets
            # (so to ensure backward compatibility)

         # The entire dataset is re-built without empty columns
        Xk_obs = [item for item in Xk if item is not np.nan]
        #Xk_obs = [item for item in Xk if type(item) is not str]
        X_obs = pd.concat(Xk_obs, axis=1)
        
        return (X_obs,Xk,ViewsX, target)

    def normalize_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        This function normalize the dataset X using min max scaler.
            :return normalized pandas dataframe norm_dataset
        """
        if self.is_multi_view:
            col_name = dataframe.columns
        else:
            col_name = [col.strip() for col in list(dataframe.columns)]
        x = dataframe.values  # returns a numpy array

        min_max_scaler = preprocessing.MinMaxScaler()
        try:
            x_scaled = min_max_scaler.fit_transform(x)
        except ValueError as value_error:
            raise ValueError(str(value_error) + "\nHint: this error can occur if headers are badly parsed"\
                             + "(eg using multiview datasests in single view mode)")
        norm_dataset = pd.DataFrame(x_scaled,
                                    index=dataframe.index,
                                    columns=col_name)
        return norm_dataset

    def initial_loc_params(self,q_i,ViewsX):
        """
        This function initializes Wk and Sigmak (randomly if no prior is provided, 
        using the global prior distribution otherwise).
            :return list of np.arrays (d_k x q) Wk
            :return list of floats > 0 Sigma2
        """
        q = self.n_components
        D_i = self.dim_views

        Wk = {}
        Sigma2 = {}
        
        for k, k_name in enumerate(self.views_iterator):
            if ViewsX[k] == 1:

                if ((self.params_dict['tilde_Wk'][k_name] is None) or (self.params_dict['sigma_til_Wk'][k_name] is None)):
                    W_k = np.random.uniform(-2,2, size = D_i[k]*q).reshape([D_i[k],q])
                else:
                    W_k = matrix_normal.rvs(mean=self.params_dict['tilde_Wk'][k_name].reshape(D_i[k], q),
                                                    rowcov=np.eye(D_i[k]),
                                                    colcov=self.params_dict['sigma_til_Wk'][k_name]*np.eye(q)).reshape(D_i[k], q)
                if q_i[k] < q:
                    W_k[:, q_i[k]:q] = 0

                if ((self.params_dict['Alpha'][k_name] is None) or (self.params_dict['Beta'][k_name] is None)):
                    s = np.random.uniform(.1,.5)
                else:
                    s = float(invgamma.rvs(a=self.params_dict['Alpha'][k_name], scale=self.params_dict['Beta'][k_name]))
                Wk[k_name] = W_k
                Sigma2[k_name] = s

            else:
                Wk[k_name] = np.nan
                Sigma2[k_name] = np.nan
                #Wk.append(np.nan)
                #Sigma2.append(np.nan)
                #Wk.append('NaN')
                #Sigma2.append('NaN')

        return Wk, Sigma2

    def EM_Optimization(self,N,q_i,Xk,Wk,Sigma2,ViewsX):
        """
        This function performs one iteration of EM (Expectation Maximization) or MAP (Maximum A Posteriori)
        optimization.
          :param N (int): number of samples
          :param q_i (list): corrected latent dimension
          :param Xk (list): list of view-specific dataset
          :param Wk (list): list of view-specific Wk parameters at previous iteration
          :param Sigma2 (list): list of view-specific sigma2k parameters at previous iteration
          :param ViewsX (list): indicator function for observed views
          :return optimized local parameters (muk, Wk, Sigma2) and updated expected LL
        """
        q = self.n_components
        D_i = self.dim_views
        # mu
        
        muk = self.eval_muk(N,Xk,Wk,Sigma2,ViewsX)
        # matreces M, B computation
        
        M, B = self.eval_MB(Wk,Sigma2,ViewsX)
        # ||tn^kg-mu^k)||2, (tn^kg-mu^k)
        norm2, tn_muk = self.compute_access_vectors(Xk, N, muk, ViewsX)

        # ================================== #
        # E-step                             #
        # ================================== #
        # evaluate mean, second moment for each x_n and expected log-likelihood
        
        E_X, E_X_2, ELL = self.compute_moments_LL(N, Sigma2, norm2, tn_muk, Wk, M, B, ViewsX)

        # ================================== #
        # M-step                             #
        # ================================== #

        #  W, Sigma2
        Wk, Sigma2_new = self.eval_Wk_Sigma2_new(N, q_i, norm2, tn_muk, E_X, E_X_2, Sigma2, ViewsX)
        # Check Sigma2_new>0
        for k,k_name in enumerate(self.views_iterator):
            if ViewsX[k] == 1:
                if Sigma2_new[k_name] > 0:
                    Sigma2[k_name] = deepcopy(Sigma2_new[k_name])
                else:
                    logger.info(f'Warning: sigma2(%i)<0 (={Sigma2_new[k_name]})' % (k + 1))

        return muk, Wk, Sigma2, ELL

    def eval_muk(self,N,Xk,Wk,Sigma2,ViewsX):
        """
        This function optimizes muk at each EM or MAP iteration step.
          :param N (int): number of samples
          :param Xk (list): list of view-specific dataset
          :param Wk (list): list of view-specific Wk parameters at previous iteration
          :param Sigma2 (list): list of view-specific sigma2k parameters at previous iteration
          :param ViewsX (list): indicator function for observed views
          :return optimized muk
        """
        D_i = self.dim_views
        muk = {}

        for k, k_name in enumerate(self.views_iterator):
            if ViewsX[k] == 1:
                if ((self.params_dict['tilde_muk'][k_name] is None) or (self.params_dict['sigma_til_muk'][k_name] is None)):
                    
                    mean_tnk = Xk[k].mean(axis=0).values.reshape(D_i[k], 1)
                    muk[k_name] = mean_tnk
                else:
                    mu_1 = np.zeros((D_i[k], 1))
                    for n in range(N):
                        mu_1 += Xk[k].iloc[n].values.reshape(D_i[k], 1)
                    term1 = self.compute_inv_term1_muck(Wk[k_name],
                                                        N, Sigma2[k_name],
                                                        self.params_dict['sigma_til_muk'][k_name])
                    Cc = self.compute_Cck(Wk[k_name], Sigma2[k_name])
                    term2 = mu_1 + (1 / self.params_dict['sigma_til_muk'][k_name]) * Cc.dot(self.params_dict['tilde_muk'][k_name].reshape(D_i[k], 1))
                    muk[k_name] = term1.dot(term2)
            else:
                muk[k_name] = np.nan
                #muk.append(np.nan)
                #muk.append('NaN')

        return muk

    def eval_MB(self, Wk, Sigma2,ViewsX):
        """
        This function evaluate matrices M and B at each EM or MAP iteration step. 
        M:=inv(I_q+sum_k Wk.TWk/sigma2k) and B:= [W1.T/sigma2K,...,W1.T/sigma2K].
        M = (sum(1/sigma2_k t(W_k)W_k) + id)^-1 
        B = [W_0/sigma2_0 --- W_d/sigma2_d]
        These matrices are needed to compute the expected LL.
          :param Wk (list): list of view-specific Wk parameters at previous iteration
          :param Sigma2 (list): list of view-specific sigma2k parameters at previous iteration
          :param ViewsX (list): indicator function for observed views
          :return matrices M and B
        """
        q = self.n_components
        D_i = self.dim_views
        index = ViewsX.index(1)  # get all views that has been specified (here get first occurence)
        # TODO: self.index has not been defined (to be solved with self.ViewsX)
        # TODO: handle case where there is only one view
        
        if self.is_multi_view:
            index_name = self.views_names[index]
        else:
            index_name = index
        # first computation of M and B
        M1 = Wk[index_name].reshape(D_i[index], q).T.dot(Wk[index_name].reshape(D_i[index],q)) / Sigma2[index_name]
        B = Wk[index_name].reshape(D_i[index], q).T / Sigma2[index_name]
        for k, k_name in zip(range(index + 1, self.K), self.views_iterator[index+1:]):
            # iterate over next computations 
            if ViewsX[k] == 1:
                # print(k,Wk[k])
                print(index_name, index, k_name)
                M1 += Wk[k_name].reshape(D_i[k], q).T.dot(Wk[k_name].reshape(D_i[k],q)) / Sigma2[k_name]
                B = np.concatenate((B, (Wk[k_name].reshape(D_i[k], q)).T / Sigma2[k_name]), axis=1)

        M = solve(np.eye(q) + M1, np.eye(q))

        return M, B

    def compute_access_vectors(self, Xk, N, muk,ViewsX):
        """
        This function computes for each subject n the vectors (tn^kg-mu^k) and the corresponding norm.

          :param Xk (list): list of view-specific dataset
          :param N (int): number of samples
          :param muk (list): list of view-specific muk parameters at previous iteration
          :param ViewsX (list): indicator function for observed views
          :return two lists containing ||tn^kg-mu^k||^2 and (tn^kg-mu^k)
        """
        D_i = self.dim_views

        norm2 = [] # norm**2 of (tn^kg-mu^k)
        tn_muk = [] # (tn^kg-mu^k)
        for n in range(N):
            norm2_k = []
            tn_mu_k = []
            for k, k_name in enumerate(self.views_iterator):
                if ViewsX[k] == 1:
                    tn_mu_k.append(Xk[k].iloc[n].values.reshape(D_i[k], 1) - muk[k_name])
                    norm2_k.append(np.linalg.norm(tn_mu_k[k]) ** 2)
                else:
                    norm2_k.append(np.nan)
                    tn_mu_k.append(np.nan)
                    #norm2_k.append('NaN')
                    #tn_mu_k.append('NaN')
            norm2.append(norm2_k)
            tn_muk.append(tn_mu_k)

        return norm2, tn_muk

    def compute_moments_LL(self, N, Sigma2, norm2, tn_muk, Wk, M, B ,ViewsX):
        """
        This function computes the first and second moments for latent variables,
        and the expected LL at each EM or MAP iteration.
          :param N (int): number of samples
          :param Sigma2 (list): list of view-specific sigma2k parameters at previous iteration
          :param norm2 (list): list of reals, ||tn^kg-mu^k||^2, (n: nb of samples,
          k: nb of features,  g: considered group)
          :param tn_muk (list): list of vectors, (tn^kg-mu^k)
          :param M (np matrix): M:=inv(I_q+sum_k Wk.TWk/sigma2k)
          :param B (np matrix): B:= [W1.T/sigma2K,...,W1.T/sigma2K]
          :param ViewsX (list): indicator function for observed views
          :return tuple(E[X], E[X^2], ) moments and expected LL
          ie first moment E[X] := M.B.(t_n - mu), second moment E[X^2] := M + E[X].t(E[X]),
          expected log-likelihood LL = - sum(dk/2 . log(sigma2) + ||tn^kg-mu^k||^2 + 1/(2.sigma2) + ...)
        """
        q = self.n_components
        D_i = self.dim_views

        E_X = []
        E_X_2 = []
        E_L_c = 0.0

        index = ViewsX.index(1)

        for n in range(N):
            tn_mu = tn_muk[n][index]
            for k in range(index+1,self.K):
                if ViewsX[k] == 1:
                    tn_mu = np.concatenate((tn_mu, tn_muk[n][k]), axis=0)
            E_X.append(M.dot(B).dot(tn_mu))

            E_X_2.append(M + E_X[n].dot(E_X[n].T))

            E_L_c_k = 0.0
            for k, k_name in enumerate(self.views_iterator):
                if ViewsX[k] == 1:
                    E_L_c_k += - D_i[k] * log(Sigma2[k_name])/ 2.0 - (norm2[n][k] / 2 + np.matrix.trace(
                        (Wk[k_name].reshape(D_i[k], q)).T.dot(Wk[k_name].reshape(D_i[k], q)) * E_X_2[n]) / 2 - E_X[n].T.dot(
                        (Wk[k_name].reshape(D_i[k], q)).T).dot(tn_muk[n][k])) / Sigma2[k_name]
            E_L_c += float(E_L_c_k  - np.matrix.trace(E_X_2[n]) / 2.0)

        return E_X, E_X_2, E_L_c

    def eval_Wk_Sigma2_new(self, N, q_i, norm2, tn_muk, E_X, E_X_2, Sigma2, ViewsX):
        """
        This function optimizes Wk and sigma2k at each EM or MAP iteration step.
          :param N (int): number of samples
          :param q_i (list): corrected latent dimension
          :param norm2 (list): list of reals, ||tn^kg-mu^k||^2
          :param tn_muk (list): list of vectors, (tn^kg-mu^k)
          :param E_X (list): list of first moments of x_n
          :param E_X_2 (list): list of second moments of x_n
          :param Sigma2 (list): list of view-specific sigma2k parameters at previous iteration
          :param ViewsX (list): indicator function for observed views
          :return optimized Wk and sigma2k
        """
        q = self.n_components
        D_i = self.dim_views

        Wk = {}
        Sigma2_new = {}

        for k, k_name in enumerate(self.views_iterator):
            if ViewsX[k] == 1:
                W_1_1 = (tn_muk[0][k]).dot(E_X[0].T)
                W_2_2 = sum(E_X_2)
                for n in range(1,N):
                    W_1_1 += (tn_muk[n][k]).dot(E_X[n].T)

                if ((self.params_dict['tilde_Wk'][k_name] is None) or (self.params_dict['sigma_til_Wk'][k_name] is None)):
                    W_1 = W_1_1

                    W_2 = solve(W_2_2, np.eye(q))
                else:
                    W_1 = W_1_1 + (Sigma2[k_name] / self.params_dict['sigma_til_Wk'][k_name]) * self.params_dict['tilde_Wk'][k_name]

                    W_2 = solve(W_2_2 + (Sigma2[k_name] / self.params_dict['sigma_til_Wk'][k_name]) * np.eye(q), np.eye(q))

                W_k = W_1.dot(W_2)
                if q_i[k] < q:
                    W_k[:, q_i[k]:q] = 0
                Wk[k_name] = W_k

                sigma2k = 0.0
                for n in range(N):
                    sigma2k += float(norm2[n][k] + np.matrix.trace(
                        (Wk[k_name].reshape(D_i[k], q)).T.dot(Wk[k_name].reshape(D_i[k], q)) * E_X_2[n]) - 2 * E_X[n].T.dot(
                        (Wk[k_name].reshape(D_i[k], q)).T).dot(tn_muk[n][k]))
                if self.params_dict['tilde_Sigma2k'][k_name] is None:
                    var = 1  # variance of the Inverse-Gamma prior
                    alpha = 1.0 / (4 * var) + 2
                    beta = (alpha - 1) / 2
                    sigma2k_N = (sigma2k + 2 * beta) / (N * D_i[k] + 2 * (alpha + 1))  ## prior=inverse gamma
                    while sigma2k_N <= 0:  # while till obtention of a non negative sigma2k: each round the variance of the Inverse Gamma is divided by 2
                        var *= 1.0 / 2
                        alpha = 1.0 / (4 * var) + 2
                        beta = (alpha - 1) / 2
                        sigma2k_N = (sigma2k + 2 * beta) / (N * D_i[k] + 2 * (alpha + 1))  ## prior=inverse gamma
                    if var != 1:
                        print(f'Variance of Inverse-Gamma for sigma2(%i) = {var}' % (k + 1))
                    Sigma2_new[k_name]= sigma2k_N
                else:
                    sigma2k_N = (sigma2k + 2 * self.params_dict['Beta'][k_name]) / \
                        (N * D_i[k] + 2 * (self.params_dict['Alpha'][k_name] + 1))
                    Sigma2_new[k_name] = sigma2k_N
            else:
                Wk[k_name] = np.nan
                Sigma2_new[k_name] = np.nan
                #Wk.append(np.nan)
                #Sigma2_new.append(np.nan)
                #Wk.append('NaN')
                #Sigma2_new.append('NaN')
        return Wk, Sigma2_new

    @staticmethod
    def compute_Cck(Wk, Sigk):
        """
        Computes matrix Ck.
        :param Wk: matrix (d_k x q)
        :param Sigk: float > 0
        :return np.array : matrix Cck
        """

        dk = Wk.shape[0]

        Cck = Wk.dot(Wk.T)+Sigk*np.eye(dk)

        return Cck

    @staticmethod
    def compute_inv_term1_muck(Wk, Nc, Sigk, Til_Sigk):
        """
        :param Wk: matrix (d_k x q)
        :param Nc: float > 0
        :param Sigk: float > 0
        :param Til_Sigk: float > 0
        :return np.array : inverse of first term to evaluate muck, using the Woodbury matrix identity
        """

        dk = Wk.shape[0]
        q = Wk.shape[1]

        k = 1.0/(Nc*Til_Sigk+Sigk)

        Inverse = solve(np.eye(q)+k*Wk.T.dot(Wk), np.eye(q))

        inverse_term1 = k*Til_Sigk*(np.eye(dk)-k*Wk.dot(Inverse).dot(Wk.T))

        return inverse_term1
