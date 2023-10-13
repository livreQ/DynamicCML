import torch
import torch.nn as nn
import torch.nn.functional as F
import  argparse
import numpy as np
import time
import math
from copy import deepcopy
from   scipy.stats import norm
from scipy.stats import multivariate_normal
from   scipy.special import logsumexp
import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm

# use gaussian to model the distribution of algorithm outputs.
class GaussianUnknownMean:                                                         
    
    def __init__(self, mean0, var0, varx):
        """Initialize model.
        
        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0  = var0
        self.varx  = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])
    
    def log_pred_prob(self, t, x):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_stds  = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)
    
    def update_params(self, t, x):
        """Upon observing a new datum x at time t, update all run length 
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        new_prec_params  = self.prec_params + (1/self.varx)
        self.prec_params = np.append([1/self.var0], new_prec_params)
        # See eq. 24 in (Murphy 2007).
        new_mean_params  = (self.mean_params * self.prec_params[:-1] + \
                            (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1./self.prec_params + self.varx

class OnlineEnvChangePointDetection:
    """
     Detect Environement Change in a Sequential way
    """

    def __init__(self, config):
        super().__init__()

        self.config = deepcopy(config)
        self.w_dim = config.paradim
        self.horizon = config.horizon
        #self.model = GaussianUnknownMean(np.zeros(self.w_dim), 2*np.ones(self.w_dim), 1*np.ones(self.w_dim))
        self.model = GaussianUnknownMean(0, 0.1, 1)
        self.hazard = config.hazard
        self.logR = -np.inf * np.ones((self.horizon + 1, self.horizon + 1))
        self.logR[0, 0] = 0
        self.pmean = np.empty(self.horizon + 1)
        self.pvar = np.empty(self.horizon + 1)
        self.log_message = np.array([0])  # log 0 == 1
        self.log_H       = np.log(self.hazard)
        self.log_1mH     = np.log(1 - self.hazard)
        self.t = 1
        self.last_decide_t=0

    def update(self, data_t):
        #print(np.shape(self.logR[]))
        # make model predictions
      
        self.pmean[self.t-1] = np.sum(np.exp(self.logR[self.t-1, :self.t]) * self.model.mean_params[:self.t])
        self.pvar[self.t-1]  = np.sum(np.exp(self.logR[self.t-1, :self.t]) * self.model.var_params[:self.t])
        # 3. Evaluate predictive probabilities.
        log_pis = self.model.log_pred_prob(self.t, data_t)
        # print("logpi", log_pis)
        # 4. Calculate growth probabilities.
        log_growth_probs = log_pis + self.log_message + self.log_1mH

        # 5. Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pis + self.log_message + self.log_H)

        # 6. Calculate evidence
        new_log_joint = np.append(log_cp_prob, log_growth_probs)
   
        # 7. Determine run length distribution.
        self.logR[self.t, :self.t+1]  = new_log_joint
        self.logR[self.t, :self.t+1] -= logsumexp(new_log_joint)

        # 8. Update sufficient statistics.
        self.model.update_params(self.t, data_t)

        # Pass message.
        self.log_message = new_log_joint

        
        self.t += 1
        return np.exp(self.logR) # at time t r=t represent the belief of current run length
    
    def decide_change_point(self, t):
        """
         Use R matrix to decide the changing point
        """
        R = np.exp(self.logR)
        # p = R[self.t, : t - self.last_decide_t + 1]
        # print(p)
        # p = p/np.sum(p)
        # return p[-1] 
        max = R.max(axis=1, keepdims=True)
        max_index = R.argmax(axis=1)
        #print(max_index)
        return max_index

    def detect(self, t):
        flag = 0
        #max_ind = np.zeros(eocd[0].horizon + 1)
        R = np.exp(self.logR)
        max_ind = R.argmax(axis=1)
        if max_ind[t+1] < max_ind[t]:# and max[t] > 0.5:
            flag =1
        else:
            flag = 0
        return flag
       
    def plot_posterior(self,  data, actual_change, detect):
        fig, axes = plt.subplots(2, 1, figsize=(8,7))
        R = np.exp(self.logR)
        ax1, ax2 = axes

        ax1.scatter(range(0, self.horizon), data)
        ax1.plot(range(0, self.horizon), data)
        ax1.set_xlim([0, self.horizon])
        ax1.margins(0)
    
        # Plot predictions.
        ax1.plot(range(0, self.horizon + 1), self.pmean, c='k')
        _2std = 2 * np.sqrt(self.pvar)
        ax1.plot(range(0, self.horizon + 1), self.pmean - _2std, c='k', ls='--')
        ax1.plot(range(0, self.horizon + 1), self.pmean + _2std, c='k', ls='--')

        ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', 
               norm=LogNorm(vmin=0.0001, vmax=1))
        ax2.set_xlim([0, self.horizon])
        ax2.margins(0)

        for cp in actual_change:
            ax1.axvline(cp, c='red', ls='dotted')
            ax2.axvline(cp, c='red', ls='dotted')
        
        for cp in detect:
            ax1.axvline(cp, c='green', ls='dotted')
            ax2.axvline(cp, c='green', ls='dotted')

        plt.tight_layout()
        plt.show()

def generate_data(varx, mean0, var0, T, cp_prob):
    """Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
    """
    data  = []
    cps   = []
    meanx = mean0
    for t in range(0, T):
        if np.random.random() < cp_prob:
            meanx = np.random.normal(mean0, var0)
            cps.append(t)
        data.append(np.random.normal(meanx, varx))
    return data, cps

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    #data settings
    argparser.add_argument('--horizon', type=int, help='horizon of tasks', default=100)
    argparser.add_argument('--paradim', type=int, help='model parameter dimension', default=1)
    argparser.add_argument('--hazard', type=int, help='hazard', default=1/10)
    argparser.add_argument('--sample_num', type=int, help='sample number per task', default=40)
    argparser.add_argument('--test_num', type=int, help='test sample number per task', default=40)
    argparser.add_argument('--sigma_eps', type=int, help='noise added on sinusoid', default=0.001)
    # loss function settings
    argparser.add_argument('--loss', type=str, help='specify loss function', default='mse')
    argparser.add_argument('--task', type=str, help='task type', default='mean_estimation')
    # base learner settings
    argparser.add_argument('--K', type=int, help='inner SGD steps', default=20)
    argparser.add_argument('--K_test', type=int, help='inner test SGD steps', default=15)
    # meta learner settings
    argparser.add_argument('--L', type=int, help='Lipschitz constant', default=20)
    argparser.add_argument('--alpha', type=int, help='alpha quadratic growth', default=2)
    argparser.add_argument('--D_hat_guess', type=int, help='D_hat initialization', default=2)
    argparser.add_argument('--rho', type=int, help='hopping learning rate', default=0.8)
    # output setting
    argparser.add_argument('--eva_bound', type=bool, help='evaluate bound or not', default=1)
    args = argparser.parse_args()
    T      = 100   # Number of observations.
    hazard = 1/10  # Constant prior on changepoint probability.
    mean0  = 0      # The prior mean on the mean parameter.
    var0   = 2      # The prior variance for mean parameter.
    varx   = 1      # The known variance of the data.

    data, cps = generate_data(varx, mean0, var0, T, hazard)
    
    oecpd = OnlineEnvChangePointDetection(args)
    detect = []
    for t in range(1, T+1):
        # 2. Observe new datum.
        x = data[t-1]
        PR = oecpd.detect(x)
        # print(PR)
        flag = oecpd.decide_change_point(t)
        print(t, flag)
        # print(Pred)
        if flag:
            detect.append(t)
    
    oecpd.plot_posterior(data, cps, detect) 