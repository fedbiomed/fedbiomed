import inspect

from joblib import dump, load

import numpy as np
import pandas as pd
from sklearn import preprocessing
from copy import deepcopy
from scipy.stats import matrix_normal, invgamma
from numpy.linalg import solve
from math import log
from fedbiomed.common.logger import logger

class PpcaPlan():
    def __init__(self,
                 kwargs):
        """
           Class initializer.
           :param global_params_dict (dictionary) containing global model parameters
           :kwargs (dictionary) containing the total number of observed views (tot_views),
                                           the dimension of each view (dim_views), 
                                           the dimension of the latent space (n_components)
        """
        # list dependencies of the model
        self.dependencies = [
                             "from fedbiomed.common.ppca import PpcaPlan",
                             "import pandas as pd"
                             ]

        self.dataset_path = None
                        
        self.K = kwargs['tot_views']
        self.dim_views = kwargs['dim_views']
        self.n_components = kwargs['n_components']
        self.norm = kwargs['norm']

        self.params_dict = {'K': self.K,
                            'dimensions': (self.dim_views,self.n_components),
                            # local params:
                            'Wk': None, 
                            'muk': None,
                            'sigma2k': None,
                            # global params:
                            'tilde_muk': [None for _ in range(self.K)],
                            'tilde_Wk': [None for _ in range(self.K)],
                            'tilde_Sigma2k': [None for _ in range(self.K)],
                            'Alpha': [None for _ in range(self.K)],
                            'Beta': [None for _ in range(self.K)],
                            'sigma_til_muk': [None for _ in range(self.K)],
                            'sigma_til_Wk': [None for _ in range(self.K)],
                            'sigma_til_sigma2k': [None for _ in range(self.K)]}

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

    #################################################
    def training_routine(self, 
                         n_iterations: int=None,
                         logger=None):

        """ 
        Args:
            norm (bool): boolean to decide if the dataset should be normalized or not. Defaults to True.
            n_iterations (int): the number of EM/MAP iterations for the current round. Defaults to None
        """

        #use_cuda = torch.cuda.is_available()
        #device = torch.device("cuda" if use_cuda else "cpu")
        #self.device = "cpu"

        # labels can be provided or not. Note that to perform optimization, 
        # only data and information on observed views should be provided.
        if len(self.training_data()) == 4:
            X, Xk, ViewsX, y = self.training_data()
        elif len(self.training_data()) == 3:
            X, Xk, ViewsX = self.training_data()

        N = X.shape[0]
        q = self.n_components
        D_i = self.dim_views

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

        # epochs indices for saving results
        sp_arr = np.arange(1, n_iterations + 1)[np.round(
            np.linspace(0, len(np.arange(1, n_iterations + 1)) - 1, int(n_iterations / 3))).astype(
            int)]

        for i in range(1, n_iterations + 1):
            muk, Wk, Sigma2, ELL = self.EM_Optimization(N,q_i,Xk,Wk,Sigma2,ViewsX)
            if i in sp_arr:
                print('Iteration: {}/{}\tExpected LL: {:.6f}'.format(i,n_iterations,ELL))

        # update local parameters
        #self.load_params({'Wk': Wk, 'muk': muk, 'sigma2k': Sigma2})
        self.update_params({'Wk': Wk, 'muk': muk, 'sigma2k': Sigma2})

    # provided by fedbiomed // necessary to save the model code into a file
    def add_dependency(self, dep):
        """
           Add new dependency to this class.
           :param dep (string) dependency to add.
        """
        self.dependencies.extend(dep)
        pass

    '''Save the code to send to nodes '''
    def save_code(self, filename):
        """Save the class code for this training plan to a file
           :param filename (string): path to the destination file
        """
        content = ""
        for s in self.dependencies:
            content += s + "\n"

        content += "\n"
        content += inspect.getsource(self.__class__)

        # try/except todo
        file = open(filename, "w")
        file.write(content)
        file.close()

    def save(self, filename, params: dict=None):
        """
        Save method for parameter communication, internally is used
        dump and load joblib library methods.
        :param filename (string)
        :param params (dictionary) model parameters to save

        Save can be called from Job or Round.
            From round is always called with params.
            From job is called with no params in constructor and
            with params in update_parameters.

            Torch state_dict has a model_params object. model_params tag
            is used in the code. This is why this tag is
            used in this case too.
        """
        file = open(filename, "wb")
        if params is None:
            dump(self.params_dict, file)
        else:
            if params.get('model_params') is not None: # called in the Round
                self.update_params(params['model_params'])
                dump(self.params_dict, file)
            else:
                self.update_params(params)
                dump(self.params_dict, file)
        file.close()

    def load(self, filename, to_params: bool = False):
        """
        Method to load the updated parameters.
        load can be called from Job or Round.
        From round is called with no params
        From job is called with params
        :param filename (string)
        :param to_params (boolean)
        :return dictionary with the loaded parameters.
        """
        di_ret = {}
        file = open( filename , "rb")
        if not to_params:
            params_dict = load(file)
            self.update_params(params_dict)
            di_ret =  params_dict
        else:
            params_dict =  load(file)
            self.update_params(params_dict)
            di_ret['model_params'] = params_dict
        file.close()
        return di_ret

    def set_dataset(self, dataset_path):
        """
          :param dataset_path (string)
        """
        self.dataset_path = dataset_path
        logger.debug('Dataset_path' + str(self.dataset_path))

    def get_model(self):
        """
            :return the model parameters
        """
        all_params = self.params_dict
        return all_params

    def training_data(self):
        """
            Perform in this method all data reading and data transformations you need.
            At the end you should provide a tuple (X_obs,Xk,ViewsX,y), where: 
            X_obs is the training dataset, 
            Xk is a list containing the k-specific dataframe if the k-view has been observed or 'NaN' otherwise,
            ViewsX is the indicator function for observed vies (ViewsX[k]=1 if view k is observed, 0 otherwise)
            y the corresponding labels.
            Note: The dataset is normalized using min max scaler if model_args['norm'] is true
            Note: labels are not used for optimization purposes, but can be useful for performance evaluation.
            :raise NotImplementedError if developer do not implement this method.
        """
        raise NotImplementedError('Training data must be implemented')

    def update_params(self,params_dict):
        """
        Update model parameters
          :param params_dict (dict)
        """
        self.params_dict.update(params_dict)

    def after_training_params(self):
        """Provide a dictionary with the federated parameters you need to aggregate
            :return the federated parameters (dictionary)
        """
        return self.params_dict

    def normalize_data(self,X):
        """
        This function normalize the dataset X using min max scaler.
            :return normalized pandas dataframe norm_dataset
        """
        col_name = [col.strip() for col in list(X.columns)]
        x = X.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        norm_dataset = pd.DataFrame(x_scaled, index=X.index, columns=col_name)
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

        Wk = []
        Sigma2 = []
        for k in range(self.K):
            if ViewsX[k] == 1:
                if ((self.params_dict['tilde_Wk'][k] is None) or (self.params_dict['sigma_til_Wk'][k] is None)):
                    W_k = np.random.uniform(-2,2, size = D_i[k]*q).reshape([D_i[k],q])
                else:
                    W_k = matrix_normal.rvs(mean=self.params_dict['tilde_Wk'][k].reshape(D_i[k], q),
                                                    rowcov=np.eye(D_i[k]),
                                                    colcov=self.params_dict['sigma_til_Wk'][k]*np.eye(q)).reshape(D_i[k], q)
                if q_i[k] < q:
                    W_k[:, q_i[k]:q] = 0

                if ((self.params_dict['Alpha'][k] is None) or (self.params_dict['Beta'][k] is None)):
                    s = np.random.uniform(.1,.5)
                else:
                    s = float(invgamma.rvs(a=self.params_dict['Alpha'][k], scale=self.params_dict['Beta'][k]))
                Wk.append(W_k)
                Sigma2.append(s)
            else:
                Wk.append('NaN')
                Sigma2.append('NaN')

        return Wk, Sigma2

    def EM_Optimization(self,N,q_i,Xk,Wk,Sigma2,ViewsX):
        """
        This function performs one iteration of EM or MAP optimization.
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
        # matreces M, B
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
        for k in range(self.K):
            if ViewsX[k] == 1:
                if Sigma2_new[k] > 0:
                    Sigma2[k] = deepcopy(Sigma2_new[k])
                else:
                    print(f'Warning: sigma2(%i)<0 (={Sigma2_new[k]})' % (k + 1))

        return muk, Wk, Sigma2, ELL

    def eval_muk(self,N,Xk,Wk,Sigma2,ViewsX):
        """
        This function optimize muk at each EM or MAP iteration step.
          :param N (int): number of samples
          :param Xk (list): list of view-specific dataset
          :param Wk (list): list of view-specific Wk parameters at previous iteration
          :param Sigma2 (list): list of view-specific sigma2k parameters at previous iteration
          :param ViewsX (list): indicator function for observed views
          :return optimized muk
        """
        D_i = self.dim_views
        muk = []

        for k in range(self.K):
            if ViewsX[k] == 1:
                if ((self.params_dict['tilde_muk'][k] is None) or (self.params_dict['sigma_til_muk'][k] is None)):
                    mean_tnk = Xk[k].mean(axis=0).values.reshape(D_i[k], 1)
                    muk.append(mean_tnk)
                else:
                    mu_1 = np.zeros((D_i[k], 1))
                    for n in range(N):
                        mu_1 += Xk[k].iloc[n].values.reshape(D_i[k], 1)
                    term1 = self.compute_inv_term1_muck(Wk[k], N, Sigma2[k], self.params_dict['sigma_til_muk'][k])
                    Cc = self.compute_Cck(Wk[k], Sigma2[k])
                    term2 = mu_1 + (1 / self.params_dict['sigma_til_muk'][k]) * Cc.dot(self.params_dict['tilde_muk'][k].reshape(D_i[k], 1))
                    muk.append(term1.dot(term2))
            else:
                muk.append('NaN')

        return muk

    def eval_MB(self, Wk, Sigma2,ViewsX):
        """
        This function evaluate matrices M and B at each EM or MAP iteration step. 
        These matrices are needed to compute the expected LL.
          :param Wk (list): list of view-specific Wk parameters at previous iteration
          :param Sigma2 (list): list of view-specific sigma2k parameters at previous iteration
          :param ViewsX (list): indicator function for observed views
          :return matrices M and B
        """
        q = self.n_components
        D_i = self.dim_views
        index = ViewsX.index(1)
        # TODO: self.index has not been defined (to be solved with self.ViewsX)
        M1 = Wk[index].reshape(D_i[index], q).T.dot(Wk[index].reshape(D_i[index],q)) / Sigma2[index]
        B = Wk[index].reshape(D_i[index], q).T / Sigma2[index]
        for k in range(index + 1, self.K):
            if ViewsX[k] == 1:
                # print(k,Wk[k])
                M1 += Wk[k].reshape(D_i[k], q).T.dot(Wk[k].reshape(D_i[k],q)) / Sigma2[k]
                B = np.concatenate((B, (Wk[k].reshape(D_i[k], q)).T / Sigma2[k]), axis=1)

        M = solve(np.eye(q) + M1,np.eye(q))

        return M, B

    def compute_access_vectors(self, Xk, N, muk,ViewsX):
        """
        This function compute for each subject n the vectors (tn^kg-mu^k) and the corresponding norm.
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
            for k in range(self.K):
                if ViewsX[k] == 1:
                    tn_mu_k.append(Xk[k].iloc[n].values.reshape(D_i[k], 1) - muk[k])
                    norm2_k.append(np.linalg.norm(tn_mu_k[k]) ** 2)
                else:
                    norm2_k.append('NaN')
                    tn_mu_k.append('NaN')
            norm2.append(norm2_k)
            tn_muk.append(tn_mu_k)

        return norm2, tn_muk

    def compute_moments_LL(self, N, Sigma2, norm2, tn_muk, Wk, M, B ,ViewsX):
        """
        This function computes the first and second moments for latent variables,
        and the expected LL at each EM or MAP iteration.
          :param N (int): number of samples
          :param Sigma2 (list): list of view-specific sigma2k parameters at previous iteration
          :param norm2 (list): list of reals, ||tn^kg-mu^k||^2
          :param tn_muk (list): list of vectors, (tn^kg-mu^k)
          :param M (np matrix)
          :param B (np matrix)
          :param ViewsX (list): indicator function for observed views
          :return moments and expected LL
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
            for k in range(self.K):
                if ViewsX[k] == 1:
                    E_L_c_k += - D_i[k] * log(Sigma2[k])/ 2.0 - (norm2[n][k] / 2 + np.matrix.trace(
                        (Wk[k].reshape(D_i[k], q)).T.dot(Wk[k].reshape(D_i[k], q)) * E_X_2[n]) / 2 - E_X[n].T.dot(
                        (Wk[k].reshape(D_i[k], q)).T).dot(tn_muk[n][k])) / Sigma2[k]
            E_L_c += float(E_L_c_k  - np.matrix.trace(E_X_2[n]) / 2.0)

        return E_X, E_X_2, E_L_c

    def eval_Wk_Sigma2_new(self, N, q_i, norm2, tn_muk, E_X, E_X_2, Sigma2, ViewsX):
        """
        This function optimize Wk and sigma2k at each EM or MAP iteration step.
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

        Wk = []
        Sigma2_new = []

        for k in range(self.K):
            if ViewsX[k] == 1:
                W_1_1 = (tn_muk[0][k]).dot(E_X[0].T)
                W_2_2 = sum(E_X_2)
                for n in range(1,N):
                    W_1_1 += (tn_muk[n][k]).dot(E_X[n].T)

                if ((self.params_dict['tilde_Wk'][k] is None) or (self.params_dict['sigma_til_Wk'][k] is None)):
                    W_1 = W_1_1

                    W_2 = solve(W_2_2, np.eye(q))
                else:
                    W_1 = W_1_1 + (Sigma2[k] / self.params_dict['sigma_til_Wk'][k]) * self.params_dict['tilde_Wk'][k]

                    W_2 = solve(W_2_2 + (Sigma2[k] / self.params_dict['sigma_til_Wk'][k]) * np.eye(q), np.eye(q))

                W_k = W_1.dot(W_2)
                if q_i[k] < q:
                    W_k[:, q_i[k]:q] = 0
                Wk.append(W_k)

                sigma2k = 0.0
                for n in range(N):
                    sigma2k += float(norm2[n][k] + np.matrix.trace(
                        (Wk[k].reshape(D_i[k], q)).T.dot(Wk[k].reshape(D_i[k], q)) * E_X_2[n]) - 2 * E_X[n].T.dot(
                        (Wk[k].reshape(D_i[k], q)).T).dot(tn_muk[n][k]))
                if self.params_dict['tilde_Sigma2k'][k] is None:
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
                    Sigma2_new.append(sigma2k_N)
                else:
                    Sigma2_new.append((sigma2k + 2 * self.params_dict['Beta'][k]) / \
                        (N * D_i[k] + 2 * (self.params_dict['Alpha'][k] + 1)))
            else:
                Wk.append('NaN')
                Sigma2_new.append('NaN')

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

