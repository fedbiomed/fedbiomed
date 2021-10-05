import inspect

import pickle

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
                 kwargs,
                 global_params_dict: dict = None):
        """
           Class initializer.
           :param global_params_dict (dictionary) containing global model parameters
        """
        # list dependencies of the model
        self.dependencies = ["import random",
                             "from sklearn.metrics import mean_absolute_error",
                             "from scipy.linalg import block_diag, svd, sqrtm",
                             "from scipy.special import digamma, polygamma, logsumexp"
                             ]

        # to be configured by setters
        # TODO: dataset should contain also information concerning available views (see training_data)
        self.dataset_path = None

        self.global_params_dict = global_params_dict

        self.params_dict = {'Wk': None, 
                            'muk': None,
                            'sigma2k': None}
                        
        self.K = kwargs['tot_views']
        self.dim_views = kwargs['dim_views']
        self.n_components = kwargs['n_components']

    #################################################
    # provided by fedbiomed
    def training_routine(self, 
                         norm: bool=True, #data preprocessing to be performed elsewhere?
                         n_iterations: int=None,
                         logger=None):

        """ 
        Args:
            ep_count (int): save the total number of local iterations performed by client id_client.
                            Defaults to None
            norm (bool): boolean to decide if the dataset should be normalized or not. Defaults to True.
            K (int): total number of different views measured across centers
            dim_views (list): list containing the dimension of each view, len(dim_views)=tot num views.
                              Defaults to None
            n_components (int): the latent dimension. Defaults to None
            n_iterations (int): the number of EM/MAP iterations for the current round. Defaults to None
        """

        #use_cuda = torch.cuda.is_available()
        #device = torch.device("cuda" if use_cuda else "cpu")
        #self.device = "cpu"

        X, y, ViewsX = self.training_data()

        if norm:
            X = self.normalize_data(X)

        N = X.shape[0]
        q = self.n_components
        D_i = self.dim_views

        self.params_dict['K'] = self.K
        self.params_dict['dimensions'] = (self.dim_views,q)

        # self.q_i contains the "effective" latent space dimension per view (at least equal to original view dimension-1)
        q_i = [] 
        for i in D_i:
            if i <= q:
                q_i.append(i - 1)
            else:
                q_i.append(q)

        # self.Xk is a list contianing the view-specific local datasets
        Xk = []
        ind = 0
        for k in range(self.K):
            if ViewsX[k] == 1:
                Xk.append(X.iloc[:, ind:ind + self.dim_views[k]])
                ind += self.dim_views[k]
            else:
                Xk.append('NaN')

        # initialize W and Sigma2
        Wk, Sigma2 = self.initial_loc_params(q,q_i,D_i,K,ViewsX)

        # epochs indices for saving results
        sp_arr = np.arange(1, n_iterations + 1)[np.round(
            np.linspace(0, len(np.arange(1, n_iterations + 1)) - 1, int(n_iterations / 3))).astype(
            int)]

        for i in range(1, n_iterations + 1):
            muk, Wk, Sigma2, ELL = self.EM_Optimization(N,q,q_i,D_i,K,Xk,Wk,Sigma2,ViewsX)
            if i in sp_arr:
                print('Iteration: {}/{}\tExpected LL: {:.6f}'.format(i,n_iterations,ELL))

        # update local parameters
        self.load_params({'Wk': muk, 'muk': Wk, 'sigma2k': Sigma2})        

    # provided by fedbiomed // necessary to save the model code into a file
    def add_dependency(self, dep):
        self.dependencies.extend(dep)
        pass

    # provider by fedbiomed
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

#    def save_code(self):
#
#        content = ""
#        for s in self.dependencies:
#            content += s + "\n"
#
#        content += "\n"
#        content += inspect.getsource(self.__class__)
#
#        # try/except todo
#        file = open("PPCA_model.py", "w")
#        file.write(content)
#        file.close()

    # provided by fedbiomed
    def save(self, filename, params: dict=None):
        if params is not None:
            return (pickle.dump(params, open(filename, "wb" )))
        else:
            return pickle.dump(self.params_dict, open(filename, "wb" ))

    # provided by fedbiomed
    def load(self, filename, to_params: bool=False):
        if to_params is True:
            return pickle.load(open(filename, "rb" ))
        else:
            return self.load_params(pickle.load(open(filename, "rb" )))

    ## provided by the fedbiomed / can be overloaded // need WORK
    #def logger(self, msg):
    #    pass

    # provided by the fedbiomed // should be moved in a DATA manipulation module
    def set_dataset(self, dataset_path):
        self.dataset_path = dataset_path
        logger.debug('Dataset_path' + str(self.dataset_path))
        #print('Dataset_path',self.dataset_path)

    # provided by the fedbiomed // should be moved in a DATA manipulation module
    def training_data(self):
        """
            Perform in this method all data reading and data transformations you need.
            At the end you should provide a couple (X,y,ViewsX), where X is the training dataset, 
            y the corresponding labels, ViewsX a list, with len(ViewsX)=K, containing 1 at position i 
            if the center dispose of data for the i-th view 0 otherwise.
            :raise NotImplementedError if developer do not implement this method.
        """
        raise NotImplementedError('Training data must be implemented')

    def load_params(self,params_dict):
        self.params_dict.update(params_dict)

    def load_global_params(self,global_par_dict):
        # global_par_dict = {'tilde_muk' :, 'tilde_Wk' :, 'tilde_Sigma2k':, 'Alpha' :, 'Beta' :, 'sigma_til_muk':, 'sigma_til_Wk':}
        self.global_params_dict = global_par_dict

    def normalize_data(self,X):
        col_name = [col.strip() for col in list(X.columns)]
        x = X.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        norm_dataset = pd.DataFrame(x_scaled, index=X.index, columns=col_name)
        return norm_dataset

    def initial_loc_params(self,q,q_i,D_i,K,ViewsX):
        """
        This function initializes Wk and Sigmak.
        :return list of np.arrays (d_k x q) Wk
        :return list of floats > 0 Sigma2
        """

        Wk = []
        Sigma2 = []
        for k in range(K):
            if ViewsX[k] == 1:
                if self.global_params is None:
                    W_k = np.random.uniform(-2,2, size = D_i[k]*q).reshape([D_i[k],q])
                    s = np.random.uniform(.1,.5)
                else:
                    W_k = matrix_normal.rvs(mean=self.global_params['tilde_Wk'][k].reshape(D_i[k], q),
                                                    rowcov=np.eye(D_i[k]),
                                                    colcov=self.global_params['sigma_til_Wk'][k]*np.eye(q)).reshape(D_i[k], q)
                    s = float(invgamma.rvs(a=self.global_params['Alpha'][k], scale=self.global_params['Beta'][k]))

                if q_i[k] < q:
                    W_k[:, q_i[k]:q] = 0
                Wk.append(W_k)
                Sigma2.append(s)
            else:
                Wk.append('NaN')
                Sigma2.append('NaN')

        return Wk, Sigma2

    def EM_Optimization(self,N,q,q_i,D_i,K,Xk,Wk,Sigma2,ViewsX):
        # mu
        muk = self.eval_muk(N,D_i,K,Xk,Wk,Sigma2,ViewsX)
        # matreces M, B
        M, B = self.eval_MB(q,D_i,K,Wk,Sigma2)
        # ||tn^kg-mu^k)||2, (tn^kg-mu^k)
        norm2, tn_muk = self.compute_access_vectors(Xk, N, D_i, K, muk)

        # ================================== #
        # E-step                             #
        # ================================== #

        # evaluate mean, second moment for each x_n and expected log-likelihood
        E_X, E_X_2, ELL = self.compute_moments_LL(N, q, D_i, K, Sigma2, norm2, tn_muk, Wk, M, B)

        # ================================== #
        # M-step                             #
        # ================================== #

        #  W, Sigma2
        Wk, Sigma2_new = self.eval_Wk_Sigma2_new(N, q, q_i, D_i, K, norm2, tn_muk, E_X, E_X_2, Sigma2)
        # Check Sigma2_new>0
        for k in range(K):
            if ViewsX[k] == 1:
                if Sigma2_new[k] > 0:
                    Sigma2[k] = deepcopy(Sigma2_new[k])
                else:
                    print(f'Warning: sigma2(%i)<0 (={Sigma2_new[k]})' % (k + 1))

        return muk, Wk, Sigma2, ELL

    def eval_muk(self,N,D_i,K,Xk,Wk,Sigma2,ViewsX):
        muk = []

        for k in range(K):
            if ViewsX[k] == 1:
                if ((self.global_params is None) or (self.global_params['tilde_muk'][k] is None)):
                    mean_tnk = Xk[k].mean(axis=0).values.reshape(D_i[k], 1)
                    muk.append(mean_tnk[k].reshape(D_i[k], 1))
                else:
                    mu_1 = np.zeros((D_i[k], 1))
                    for n in range(N):
                        mu_1 += Xk[k].iloc[n].values.reshape(D_i[k], 1)
                    term1 = self.compute_inv_term1_muck(Wk[k], N, Sigma2[k], self.global_params_dict['sigma_til_muk'][k])
                    Cc = self.compute_Cck(Wk[k], Sigma2[k])
                    term2 = mu_1 + (1 / self.global_params_dict['sigma_til_muk'][k]) * Cc.dot(self.global_params_dict['tilde_muk'][k].reshape(D_i[k], 1))
                    muk.append(term1.dot(term2))
            else:
                muk.append('NaN')

        return muk

    def eval_MB(self, q, D_i, K, Wk, Sigma2,ViewsX):
        index = ViewsX.index(1)
        # TODO: self.index has not been defined (to be solved with self.ViewsX)
        M1 = Wk[index].reshape(D_i[index], q).T.dot(Wk[index].reshape(D_i[index],q)) / Sigma2[index]
        B = Wk[index].reshape(D_i[index], q).T / Sigma2[index]
        for k in range(index + 1, K):
            if ViewsX[k] == 1:
                # print(k,Wk[k])
                M1 += Wk[k].reshape(D_i[k], q).T.dot(Wk[k].reshape(D_i[k],q)) / Sigma2[k]
                B = np.concatenate((B, (Wk[k].reshape(D_i[k], q)).T / Sigma2[k]), axis=1)

        M = solve(np.eye(q) + M1,np.eye(q))

        return M, B

    def compute_access_vectors(self, Xk, N, D_i, K, muk,ViewsX):

        norm2 = [] # norm**2 of (tn^kg-mu^k)
        tn_muk = [] # (tn^kg-mu^k)
        for n in range(N):
            norm2_k = []
            tn_mu_k = []
            for k in range(K):
                if ViewsX[k] == 1:
                    tn_mu_k.append(Xk[k].iloc[n].values.reshape(D_i[k], 1) - muk[k])
                    norm2_k.append(np.linalg.norm(tn_mu_k[k]) ** 2)
                else:
                    norm2_k.append('NaN')
                    tn_mu_k.append('NaN')
            norm2.append(norm2_k)
            tn_muk.append(tn_mu_k)

        return norm2, tn_muk

    def compute_moments_LL(self, N, q, D_i, K, Sigma2, norm2, tn_muk, Wk, M, B,ViewsX):

        E_X = []
        E_X_2 = []
        E_L_c = 0.0

        index = ViewsX.index(1)

        for n in range(N):
            tn_mu = tn_muk[n][index]
            for k in range(index+1,K):
                if ViewsX[k] == 1:
                    tn_mu = np.concatenate((tn_mu, tn_muk[n][k]), axis=0)
            E_X.append(M.dot(B).dot(tn_mu))

            E_X_2.append(M + E_X[n].dot(E_X[n].T))

            E_L_c_k = 0.0
            for k in range(K):
                if ViewsX[k] == 1:
                    E_L_c_k += - D_i[k] * log(Sigma2[k])/ 2.0 - (norm2[n][k] / 2 + np.matrix.trace(
                        (Wk[k].reshape(D_i[k], q)).T.dot(Wk[k].reshape(D_i[k], q)) * E_X_2[n]) / 2 - E_X[n].T.dot(
                        (Wk[k].reshape(D_i[k], q)).T).dot(tn_muk[n][k])) / Sigma2[k]
            E_L_c += float(E_L_c_k  - np.matrix.trace(E_X_2[n]) / 2.0)

        return E_X, E_X_2, E_L_c

    def eval_Wk_Sigma2_new(self, N, q, q_i, D_i, K, norm2, tn_muk, E_X, E_X_2, Sigma2,ViewsX):

        Wk = []
        Sigma2_new = []

        for k in range(K):
            if ViewsX[k] == 1:
                W_1_1 = (tn_muk[0][k]).dot(E_X[0].T)
                W_2_2 = sum(E_X_2)
                for n in range(1,N):
                    W_1_1 += (tn_muk[n][k]).dot(E_X[n].T)

                if ((self.global_params is None) or (self.global_params['tilde_Wk'][k] is None)):
                    W_1 = W_1_1

                    W_2 = solve(W_2_2, np.eye(q))
                else:
                    W_1 = W_1_1 + (Sigma2[k] / self.global_params['sigma_til_Wk'][k]) * self.global_params['tilde_Wk'][k]

                    W_2 = solve(W_2_2 + (Sigma2[k] / self.global_params['sigma_til_Wk'][k]) * np.eye(q), np.eye(q))

                W_k = W_1.dot(W_2)
                if q_i[k] < q:
                    W_k[:, q_i[k]:q] = 0
                Wk.append(W_k)

                sigma2k = 0.0
                for n in range(N):
                    sigma2k += float(norm2[n][k] + np.matrix.trace(
                        (Wk[k].reshape(D_i[k], q)).T.dot(Wk[k].reshape(D_i[k], q)) * E_X_2[n]) - 2 * E_X[n].T.dot(
                        (Wk[k].reshape(D_i[k], q)).T).dot(tn_muk[n][k]))
                if ((self.global_params is None) or (self.global_params['tilde_Sigma2k'][k] is None)):
                    var = 1  # variance of the Inverse-Gamma prior
                    alpha = 1.0 / (4 * var) + 2
                    beta = (alpha - 1) / 2
                    sigma2k_N = (sigma2k + 2 * beta) / (N * D_i[k] + 2 * (alpha + 1))  ## prior=inverse gamma
                    while sigma2k_N <= 0:  # while til obtention of a non negative sigma2k: each round the variance of the Inverse Gamma is divided by 2
                        var *= 1.0 / 2
                        alpha = 1.0 / (4 * var) + 2
                        beta = (alpha - 1) / 2
                        sigma2k_N = (sigma2k + 2 * beta) / (N * D_i[k] + 2 * (alpha + 1))  ## prior=inverse gamma
                    if var != 1:
                        print(f'Variance of Inverse-Gamma for sigma2(%i) = {var}' % (k + 1))
                    Sigma2_new.append(sigma2k_N)
                else:
                    Sigma2_new.append((sigma2k + 2 * self.global_params['Beta'][k]) / \
                        (N * D_i[k] + 2 * (self.global_params['Alpha'][k] + 1)))
            else:
                Wk.append('NaN')
                Sigma2_new.append('NaN')

        return Wk, Sigma2_new

    @staticmethod
    def eval_mean_view(n_views,dim_views,ViewsX,Xk):
        # TODO: should figure out how information concerning available views is passed
        mean_tnk = []
        for k in range(n_views):
            if ViewsX[k] == 1:
                mean_tnk.append(Xk[k].mean(axis=0).values.reshape(dim_views[k], 1))
            else:
                mean_tnk.append(0)
        return mean_tnk

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

