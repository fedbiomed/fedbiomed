from typing import Dict
import numpy as np
from math import log
from scipy.special import digamma, polygamma

from fedbiomed.researcher.aggregators.aggregator import Aggregator


class MLaggregator(Aggregator):
    """ Defines the maximum likelihood aggregation strategy for PPCA """

    def __init__(self):
        super(MLaggregator, self).__init__()

    def aggregate(self, model_params: list, weights: list=None) -> Dict:
        """aggregates  local models sent by participating nodes into
        a global model, following Federated Averaging strategy.

        Args:
            model_params (list): contains each model layers
            weights (list): contains all weigths of a given
            layer (note: weights are not used here).

        Returns:
            Dict: [description]
        """
        assert len(model_params) > 0, 'An empty list of models was passed.'

        K = model_params[0]['K']
        dim_views,q = model_params[0]['dimensions']
        for model in model_params:
            assert model['K']==K
            d1,q1 = model['dimensions']
            assert d1 == dim_views
            assert q1 == q

        corr_det_inv = 1e-20
        rho = 1e-4

        # evaluate total number of centers Ensure compatibility with datahaving measurements for each view
        Tot_C_k_W, Tot_C_k_mu, Tot_C_k_S = self.count_participating_clients(model_params,K)

        # ================================== #
        #           ML OPTIMIZATION          #
        # ================================== #
        
        # obtain model parameters through Maximum likelihood estimation
        tilde_muk, tilde_Wk, tilde_Sigma2k, sigma_til_muk, sigma_til_Wk = \
            self.eval_gauss_global_params(model_params,
                                          K,
                                          dim_views,
                                          q,
                                          Tot_C_k_W,
                                          Tot_C_k_mu,
                                          Tot_C_k_S,
                                          corr_det_inv,
                                          rho)
        Alpha, Beta, sigma_til_sigma2k = self.eval_inv_gamma(model_params,
                                                             K,
                                                             Tot_C_k_S,
                                                             tilde_Sigma2k,
                                                             corr_det_inv,
                                                             rho)

        global_params_dict = {'tilde_muk': tilde_muk,
                              'tilde_Wk': tilde_Wk,
                              'tilde_Sigma2k': tilde_Sigma2k,
                              'Alpha': Alpha,
                              'Beta': Beta,
                              'sigma_til_muk': sigma_til_muk,
                              'sigma_til_Wk': sigma_til_Wk,
                              'sigma_til_sigma2k': sigma_til_sigma2k
                              }

        return global_params_dict

    @staticmethod
    def eval_gauss_global_params(model_params,
                                 K,
                                 D_i,
                                 q,
                                 Tot_C_k_W,
                                 Tot_C_k_mu,
                                 Tot_C_k_S,
                                 corr_det_inv,
                                 rho):
        """
        This function performs ML estimation for normally distributed parameters
        :return lists of np.arrays (tilde_muk, tilde_Wk)
        :return lists of floats (tilde_Sigma2k, sigma_til_muk, sigma_til_Wk)
        """
        # ML estimator for gaussian parameters and mean of sigma2

        tilde_muk = []
        tilde_Wk = []
        tilde_Sigma2k = []
        sigma_til_muk = []
        sigma_til_Wk = []
        for k in range(K):
            tilmuk = np.zeros((D_i[k], 1))
            tilWk = np.zeros((D_i[k], q))
            tilSk = 0.0
            for model in model_params:
                #if type(model['muk'][k]) is not str:
                if not np.isnan(model['muk'][k]).any():
                    tilmuk+=model['muk'][k]
                #if type(model['Wk'][k]) is not str:  
                if not np.isnan(model['Wk'][k]).any():
                    tilWk += model['Wk'][k]
                #if type(model['sigma2k'][k]) is not str:  
                if not np.isnan(model['sigma2k'][k]).any():
                    tilSk += model['sigma2k'][k]
            
            if Tot_C_k_S[k] >= 1:
                tilde_Sigma2k.append(tilSk / Tot_C_k_S[k])

            if Tot_C_k_W[k] > 1:
                tilde_Wk.append(1.0 / Tot_C_k_W[k] * tilWk)
                sigWk = 0.0
                for model in model_params:
                    if type(model['Wk'][k]) is not str:  # not np.isnan(model['Wk'][k]).any():
                        sigWk += np.matrix.trace((model['Wk'][k] - tilde_Wk[k]).T.dot(model['Wk'][k] - tilde_Wk[k]))
                if sigWk == 0.0:
                    sigma_til_Wk.append(corr_det_inv)
                else:
                    sigma_til_Wk_temp = sigWk / (Tot_C_k_W[k] * D_i[k] * q)
                    sigma_til_Wk.append(sigma_til_Wk_temp*min(1,np.linalg.norm(tilde_Wk[k])/(5*sigma_til_Wk_temp)))
            elif Tot_C_k_W[k] == 1:
                tilde_Wk.append(1.0 / Tot_C_k_W[k] * tilWk)
                sigma_til_Wk.append(rho)

            if Tot_C_k_mu[k] > 1:
                tilde_muk.append(1.0/Tot_C_k_mu[k]*tilmuk)
                sigmuk = 0.0
                for model in model_params:
                    #if type(model['muk'][k]) is not str:  
                    if not np.isnan(model['muk'][k]).any():
                        sigmuk+=float((model['muk'][k]-tilde_muk[k]).T.dot(model['muk'][k]-tilde_muk[k]))
                if sigmuk == 0.0:
                    sigma_til_muk.append(corr_det_inv)
                else:
                    sigma_til_muk.append(sigmuk/(Tot_C_k_mu[k]*D_i[k]))
            elif Tot_C_k_mu[k] == 1:
                tilde_muk.append(1.0/Tot_C_k_mu[k]*tilmuk)
                sigma_til_muk.append(rho)

        return tilde_muk, tilde_Wk, tilde_Sigma2k, sigma_til_muk, sigma_til_Wk

    def eval_inv_gamma(self,model_params,K,Tot_C_k_S,tilde_Sigma2k,corr_det_inv,rho):
        """
        This function performs ML estimation for inverse gamma parameters
        :return lists of floats 
        """
        # Alpha, Beta ML method

        Alpha = []
        Beta = []
        sigma_til_sigma2k = []
        for k in range(K):
            if Tot_C_k_S[k] >= 1:
                Ck_1 = 0.0
                Ck_2 = 0.0
                varSk = 0.0
                for model in model_params:
                    
                    #if type(model['sigma2k'][k]) is not str:  
                    if not np.isnan(model['sigma2k'][k]).any():
                        Ck_1 += 1.0 / model['sigma2k'][k]
                        Ck_2 += log(model['sigma2k'][k])
                        varSk += (model['sigma2k'][k] - tilde_Sigma2k[k]) ** 2
                Ck = -log(Ck_1) - Ck_2 / Tot_C_k_S[k]
                if varSk == 0.0:
                    varSk = corr_det_inv
                if Tot_C_k_S[k] == 1:
                    alphak = (tilde_Sigma2k[k] ** 2) / (varSk) + 2
                else:
                    alphak = (tilde_Sigma2k[k] ** 2) / (varSk / (Tot_C_k_S[k] - 1.0)) + 2
                for cov in range(10):
                    alphak = self.inv_digamma(y=log(Tot_C_k_S[k]*alphak)+Ck)
                if alphak<=2:
                    alphak = 2+1e-5
                betak = (Tot_C_k_S[k] * alphak) / Ck_1
                var_sigmak = betak ** 2 / (((alphak - 1) ** 2) * (alphak - 2))
                Beta.append(betak)
                Alpha.append(alphak)
                sigma_til_sigma2k.append(var_sigmak)

        return Alpha, Beta, sigma_til_sigma2k

    @staticmethod
    def count_participating_clients(model_params,K):
        """
        This function evaluates the effective number of participating clients
        per parameter for the current round
        :Params model_params: contains each model parameter from each node
        :K views: number of features for ech views
        :return tuple of lists of integers
        """

        Tot_C_k_W = []  # total of W reconstruction matrix
        Tot_C_k_mu = []  #
        Tot_C_k_S = []

        for k in range(K):
            TotCkW = 0
            TotCkmu = 0
            TotCkS = 0
            for model in model_params:
                #if type(model['Wk'][k]) is not str:  
                if not np.isnan(model['Wk'][k]).any():
                    TotCkW += 1
                #if type(model['muk'][k]) is not str:  
                if not np.isnan(model['muk'][k]).any():
                    TotCkmu += 1
                #if type(model['sigma2k'][k]) is not str: 
                if not np.isnan(model['sigma2k'][k]).any():
                    TotCkS += 1
            Tot_C_k_W.append(TotCkW)
            Tot_C_k_mu.append(TotCkmu)
            Tot_C_k_S.append(TotCkS)

        return Tot_C_k_W, Tot_C_k_mu, Tot_C_k_S

    @staticmethod
    def inv_digamma(y, eps=1e-8, max_iter=100):
        """
        Computes Numerical inverse to the digamma function by root finding.
        :return float
        """
        '''Numerical inverse to the digamma function by root finding'''

        if y >= -2.22:
            xold = np.exp(y) + 0.5
        else:
            xold = -1 / (y - digamma(1))

        for _ in range(max_iter):

            xnew = xold - (digamma(xold) - y) / polygamma(1, xold)

            if np.abs(xold - xnew) < eps:
                break

            xold = xnew

        return xnew
