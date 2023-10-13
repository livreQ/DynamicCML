#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Thu May  5 10:49:29 2022
# File Name: meta_learner.py
# Description:
"""

from __future__ import absolute_import,division, print_function
from ast import arg
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from torch import linalg as LA
from numpy import linalg as nLA
from    torch import optim
import  numpy as np
from    src.learners.model import Model
from    copy import deepcopy
from torch import nn, autograd as ag
import logging
from tqdm import tqdm
from src.utils.utils import tensors_to_device, compute_accuracy
from src.envdetector.env_change_detection import OnlineEnvChangePointDetection
#import sys

# from torchviz import make_dot
torch.manual_seed(168)
torch.cuda.manual_seed_all(168)
np.random.seed(168)

class DMOGD(nn.Module):
    """
        Dynamic Meta Online Gradient Descent
    """

    def __init__(self, args, config, model=None):
        """
        :param args:
        """
        super(DMOGD, self).__init__()

        # define model
        self.device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu"))
        if model is None:
            self.model = Model(config)
        else:
            self.model = model
        self.model = self.model.to(device=self.device)
        #self.meta_optim = optim.SGD(self.model.parameters(), lr=self.meta_lr)
        self.loss_type = args.loss
        self.task_type = args.task #'regression'#'classification', 'mean_estimation'
        self.task_horizon = args.timesteps#args.horizon# T
        #other settings
        self.eva_bound = args.eva_bound
        # some loss function related constants
        self.L = args.L#20. # Gaussian is truncted in [-10, 10]
        if args.dataset in  ['omniglot', 'tiered-imagenet', "synbols"]:
            self.m = args.n_shots * args.n_ways#20.
        else:
            self.m = args.n_shots
        self.m_test = args.n_shots_test
        self.alpha = args.alpha#2. # square loss 2-strong convex => 2-qudratic growth
        self.a = 1/2.
        self.b = self.alpha
        #==================
        self.n = 0# env counts
        self.t = 0# task counts
        #==================
        self.base_algo = args.base_algo
        self.meta_algo = args.meta_algo
        self.phase = None# pretrain/pretrain-test/continual
        #======================
        # base learner setting, mini-batch SGD
        self.inner_batch_size = args.inner_batch_size
        self.K = args.K #K epochs
        # assumptions
        self.D_hat_guess = args.D_hat_guess#
        self.rho = args.rho
        self.eta_0 = args.eta_0
        #self.gamma_0 = args.gamma_0
        self.epsilon_0_rate = args.epsilon_0_rate
        #======================
        self._update_algo_params()
        if args.pretrain:
            # pretrain phase, use predefined learning rates, non-adaptive
            self.meta_lr = args.meta_lr
            self.base_lr = args.step_size
        #======================
        self._init_evaluation_params()
        #======================
        self._init_env_change_detection_params(args)
        self.close_deepcopy = False
        self.lazy_update = False


    def _update_algo_params(self):
        """
            hyper params, self.m may change w.r.t the dataloader, need to be updated when see the data
        """
        #!!!!!!!!!!!!!!!!!!!! caused a bug !!!!!!!!!!!!!!!
        #self.inner_batch_size = min(self.m, self.inner_batch_size)
        K = np.ceil(self.K * self.m / self.inner_batch_size)
        self.kappa = np.sqrt(1/( K * self.alpha) + 2/(self.m * self.alpha )) * self.L
        self.epsilon = 2 * (self.kappa **2)
        self.delta = 2 * self.kappa
        #self.epsilon_0 = np.power(self.task_horizon, -1/4.)
        #print("========= update CL params ================")
        # =======  base learner setting, SGD ========
        if self.t == 0 or (self.t == 1 and self.phase == "continual"):
            self.epsilon_0 = self.epsilon_0_rate * np.power(self.t + 1, -1/4.)
            self.eta_0 = self.bound_eta_prime(self.eta_0) #np.sqrt((self.epsilon_0)/self.a)
            self.eta_prime = self.eta_0
            # base learner learning rate for the first task encountered
            self.base_lr = (self.eta_prime/self.kappa + 2)/( K * self.alpha)#eta
            # ======= meta learner setting, Dynamic OGD ==============
            # gamma0 =  D/G * kappa, gamma = gamma0/\sqrt{t}
            self.gamma_0 = self._cal_gamma0()
            self.gamma = self.gamma_0
            self.meta_lr = 2 * self.b * self.gamma / self.eta_0
            self.hopping_lr = self.rho/(2*self.b*self.eta_0)#args.rho#\rho
            print("sample numbers per task:", self.m)
            print("eta_0 range:", np.sqrt(self.epsilon_0/self.a), np.sqrt((self.b*self.D_hat_guess**2 + self.epsilon + self.epsilon_0)/self.a))
            self.show_learner_params()
        else:
            self.epsilon_0 = self.epsilon_0_rate * np.power(self.t, -1/4.)
        #sys.stdout.flush()

    def _cal_gamma0(self, D_hat_guess=None):
        if D_hat_guess is None:
            D_hat_guess = self.D_hat_guess
        gamma_0 = self.epsilon_0 * np.sqrt(((1 + self.b/self.a) * D_hat_guess**2 + self.epsilon/self.a)\
                /(4 * self.a * self.b**2 * D_hat_guess**2 * self.epsilon_0 + self.a**2 * (self.b * D_hat_guess**2 + self.epsilon)**2))
        return gamma_0

    def _init_evaluation_params(self):
        """
            params for estimating the Diameters, Variances of output hypothesis space in each environment
        """
        self.M_n = []# number of tasks per static env
        self.D_hat = []
        self.g_D_hat = 0
        self.w_min_per_cordinate = None
        self.w_max_per_cordinate = None
        # estimate D_hat
        self.g_w_min_per_cordinate = None
        self.g_w_max_per_cordinate = None
        self.V = []
        self.path_length = 0
        # slot mean
        self.mean_ = None
        self.last_mean_ = None
        # global mean, variance
        self.g_mean_ = None
        self.g_V = 0

        self.env_shift_measure = None
        self.ave_test_loss = 0
        self.ave_train_loss = 0
        self.static_bound = 0
        self.dynamic_bound = 0
        self.estimation = {}
        self.params = {}
        self.D_constant = 0
        self.S_constant = 0

    def _init_env_change_detection_params(self, args):
        methods = ['oracle', 'bocd', 'window', 'static']
        # env
        self.env_change = 0#False#switch_times[self.t]
        self.act_change = 0
        # env change detection methods
        self.detect = methods[args.switch_method - 1]
        self.win = args.win
        if self.detect == 'bocd':
            self.detector = OnlineEnvChangePointDetection(args)

    def _loss(self, input_, target_):
        if self.loss_type == 'cross_entropy':
            return F.cross_entropy(input_, target_, reduce='mean')
        elif self.loss_type == 'mse':
            return F.mse_loss(input_, target_)
        else:
            raise NotImplementedError


    def show_learner_params(self):
        print("==========================")
        print("Task time:", self.t)
        print("Running Phase:", self.phase)
        print("Cost function Constants, kappa:{},epsilon:{}, delta:{}, a:{},b:{},epsilon_0:{}"\
                .format(self.kappa, self.epsilon,self.delta, self.a, self.b, self.epsilon_0))
        print("Base Learner params, eta_0:{}, eta_prime:{}, base_lr:{}, update steps:{}, sample_num:{}, batch_size:{}"\
                .format(self.eta_0, self.eta_prime, self.base_lr, np.ceil(self.K * self.m /self.inner_batch_size), self.m, self.inner_batch_size))
        print("Inner loss constants, L:{}, alpha:{}".format(self.L, self.alpha))
        print("Meta Learner params, gamma_0:{}, gamma:{},rho:{}, meta_lr:{}, hopping_lr:{}".format(self.gamma_0, self.gamma, self.rho, self.meta_lr, self.hopping_lr))
        #if self.t > 1:
        #    print("Bound related estimation, D_hat_guess:{}, last_D_hat:{}, D_hat_global:{}, last_M_n:{}, last_V_n:{}, Path_length:{}, global variance:{}"\
        #        .format(self.D_hat_guess, self.D_hat[-1], self.g_D_hat, self.M_n[-1], self.V[-1], self.path_length, self.g_V))
            #print("Bound estimation, D_hat_guess:{}, D_hat:{}, D_hat_global:{}, M_n:{}, V_n:{}, Path_length:{}, global variance:{}"\
            #    .format(self.D_hat_guess, self.D_hat, self.g_D_hat, self.M_n, self.V, self.path_length, self.g_V))
        #sys.stdout.flush()
        # print("Bound estimation, D_hat_guess:{}, D_hat:{}, M_n:{}, V_n:{}, Path_length:{}, global mean:{}, global variance:{}"\
        #     .format(self.D_hat_guess, self.D_hat, self.M_n, self.V, self.path_length, self.g_mean_, self.g_V))

    def cal_grad_norm(self, grad):
        para_norm = 0
        for g in grad:
            para_norm += g.data.norm(2).item() ** 2
            #print(para_norm)
        return para_norm

    def update_min_max_cordinate(self, w_t):
        """
           maintain the min, max cordinate to estimate the diameter
        """
        # first task or environment change
        if self.w_max_per_cordinate is None or self.env_change:
            self.w_max_per_cordinate = []
            self.w_min_per_cordinate = []
            if not self.env_change or self.g_w_max_per_cordinate is None:
                self.g_w_max_per_cordinate = []
                self.g_w_min_per_cordinate = []
            for name in w_t:
                if 'bn' not in name:
                    w_tmp = deepcopy(w_t[name].view(-1))
                    origin = torch.zeros_like(w_tmp)
                    self.w_max_per_cordinate.append(torch.maximum(w_tmp, origin))
                    self.w_min_per_cordinate.append(torch.minimum(w_tmp, origin))
                    self.g_w_max_per_cordinate.append(torch.maximum(w_tmp, origin))
                    self.g_w_min_per_cordinate.append(torch.minimum(w_tmp, origin))
        else:
            # print("update")
            # print(self.w_max_per_cordinate, self.w_min_per_cordinate)
            i = 0
            for name in w_t:
                if 'bn' not in name:
                    self.w_max_per_cordinate[i] = torch.maximum(self.w_max_per_cordinate[i], deepcopy(w_t[name]).view(-1))
                    self.w_min_per_cordinate[i] = torch.minimum(self.w_min_per_cordinate[i], deepcopy(w_t[name]).view(-1))
                    self.g_w_max_per_cordinate[i] = torch.maximum(self.g_w_max_per_cordinate[i], deepcopy(w_t[name]).view(-1))
                    self.g_w_min_per_cordinate[i] = torch.minimum(self.g_w_min_per_cordinate[i], deepcopy(w_t[name]).view(-1))
                    i += 1

    def cal_current_D_hat(self):
        D_hat = 0
        for x_min, x_max in zip(self.w_min_per_cordinate, self.w_max_per_cordinate):
            D_hat += LA.norm(x_max - x_min, 2)
        if D_hat == 0:
            for x_min, x_max in zip(self.w_min_per_cordinate, self.w_max_per_cordinate):
                D_hat += LA.norm(x_max, 2)
        return D_hat.cpu().detach().numpy()

    def cal_global_D_hat(self):
        self.g_D_hat = 0
        for x_min, x_max in zip(self.g_w_min_per_cordinate, self.g_w_max_per_cordinate):
            self.g_D_hat += LA.norm(x_max - x_min, 2)
        return self.g_D_hat.cpu().detach().numpy()

    def update_mean_variance(self, w_t):
        if self.g_mean_  is None:
            self.g_mean_  = []
            for name in w_t:
                if 'bn' not in name:
                    self.g_mean_.append(deepcopy(w_t[name].view(-1)))

        if self.mean_ is None or self.env_change:
            """
                update hindsight mean
            """
            self.update_path_length()
            self.last_mean_ = self.mean_
            self.mean_ = []
            self.V.append(0)
            for name in w_t:
                if 'bn' not in name:
                    self.mean_.append(deepcopy(w_t[name].view(-1)))
        else:
            i = 0
            for name in w_t:
                if 'bn' not in name:
                    # print(w_t, self.mean_)
                    tmp1 = w_t[name].view(-1) - self.mean_[i]
                    self.mean_[i] = (1 - 1./self.M_n[-1])*self.mean_[i] + 1./self.M_n[-1]*w_t[name].view(-1)
                    tmp2 = w_t[name].view(-1) - self.mean_[i]
                    tmp3 =  w_t[name].view(-1) - self.g_mean_[i]
                    self.g_mean_[i] =  (1 - 1./self.t)*self.g_mean_[i] + 1./self.t*w_t[name].view(-1)
                    tmp4 = w_t[name].view(-1) - self.g_mean_[i]
                    self.g_V = (1 - 1./self.t)*self.g_V + torch.tensordot(tmp3, tmp4, 1).cpu().detach().numpy()/self.t
                    # print("tmp")
                    # print(tmp1, tmp2)
                    self.V[-1] = (1 - 1./self.M_n[-1])*self.V[-1] + torch.tensordot(tmp1, tmp2,1).cpu().detach().numpy()/self.M_n[-1]
                    i += 1

    def cal_param_square_distance(self, phi, w):
        dist = 0
        for name in w:
            if 'bn' not in name:
                dist += torch.sqrt(torch.sum((w[name].view(-1) - phi[name].view(-1)) ** 2) + 1e-10) **2
            #LA.norm(w[name].view(-1) - phi[name].view(-1), 2)**2
        # if torch.isnan(dist):
        #     print(phi, w)
        return dist

    def cal_param_norm(self, w):
        dist = 0
        for name in w:
            if 'bn' not in name:
                dist += torch.sqrt(torch.sum((w[name].view(-1)) ** 2) + 1e-10) **2
        return dist

    def update_path_length(self):
        if self.last_mean_ is not None:
            for x, y in zip(self.last_mean_, self.mean_):
                self.path_length += LA.norm(x - y, 2).cpu().detach().numpy() \
                    + np.abs(np.sqrt((self.b*self.V[-1] + self.epsilon + self.epsilon_0)/self.a)\
                         - np.sqrt((self.b*self.V[-2] + self.epsilon + self.epsilon_0)/self.a))

    def bound_eta_prime(self, eta_prime):
        L, U = np.sqrt(self.epsilon_0/self.a), np.sqrt((self.b*self.D_hat_guess**2 + self.epsilon + self.epsilon_0)/self.a)
        if eta_prime is not None or torch.isnan(eta_prime):
            if eta_prime > U:
                print(eta_prime, "eta_prime exeeds range:", U)
                eta_prime = U
            if eta_prime < L:
                print(eta_prime, "eta_prime exeeds range:", L)
                eta_prime = L
            if np.isnan(eta_prime):
                print(eta_prime, "eta_prime is nan")
                eta_prime = U
        return eta_prime

    def totorch(self, x):
        return ag.Variable(torch.Tensor(x))

    def _interpolate_model_params(self, previous_model, current_model, p, ignore_bn_exd=True):
        #else:
        #w_average = deepcopy(self._interpolate_model_params(w_average, 1./(task_idx + 1)))
        #w_average = {name : (1 - 1./(task_idx + 1)) * w_average[name] \
        #                           + 1./(1 + task_idx) * self.model.state_dict()[name] for name in self.model.state_dict()}
        for name in current_model:
            if ignore_bn_exd:
                if 'bn' not in name:
                    previous_model[name] = (1 - p) * previous_model[name] + p * current_model[name]
                else:
                    previous_model[name] = current_model[name]
            else:
                previous_model[name] = (1 - p) * previous_model[name] + p * current_model[name]
        return previous_model

    def SGD(self, x_train, y_train, x_test, y_test, test=False):
        """
            SGD as base learner
        """
        with torch.no_grad():
           train_out = self.model(x_train)
        inds = np.random.permutation(len(x_train))
        for k in range(self.K):
            #mini batch
            for start in range(0, len(x_train), self.inner_batch_size):# use m_t example K steps
                mbinds = inds[start:start + self.inner_batch_size]
                x, y = x_train[mbinds], y_train[mbinds]
                ypred = self.model(x)
                loss = self._loss(ypred, y)
                #print(loss.cpu().detach().numpy())
                self.model.zero_grad()
                loss.backward()# compute gradient, do not update param
                #torch.autograd.grad(loss, self.model.parameters())
                for param in self.model.parameters():
                    param.data -= self.base_lr * param.grad.data # sgd
                # print("param data", param.data)

        with torch.no_grad():
            #train_out = self.model(x_train)
            train_loss = self._loss(train_out, y_train)
            train_loss = train_loss.cpu().detach().numpy()
            self.ave_train_loss = ((self.t - 1)*self.ave_train_loss + train_loss)/self.t
            if test:
                # use fast weights w_t
                test_out = self.model(x_test)
                test_loss = self._loss(test_out, y_test)
                test_loss = test_loss.cpu().detach().numpy()
                self.ave_test_loss = ((self.t - 1)*self.ave_test_loss + test_loss)/self.t
            else:
                test_loss = None

        if self.task_type == 'classification':

            train_performance = compute_accuracy(train_out, y_train)
            self.env_shift_measure = train_performance
            if test:
                test_performance = compute_accuracy(test_out, y_test) #np.array(correct) / float(self.m_test)

            else:
                test_performance = None
        elif self.task_type == 'mean_estimation':
            #print(self.model.parameters())
            train_performance = deepcopy(self.model.parameters())[0].cpu().detach().numpy()#list(map(lambda p: p, self.model.parameters()))[0].cpu().detach().numpy()
            self.env_shift_measure = train_performance
            test_performance = None
        else:
            train_performance = train_out.cpu().detach().numpy()#loss before inner update: meta model performed on new task data
            if test:
                test_performance = test_out.cpu().detach().numpy()
            else:
                test_performance = None
        return train_performance, test_performance, train_loss, test_loss


    def base_learner(self, data, test=False):
        """
            We can implement more types of base learners, batch of tasks
        """
        # update meta-params w.r.t task batches

        num_tasks = data['train'][0].size(0)
        results = {'num_tasks': num_tasks,
                    'inner_losses': np.zeros((self.K, num_tasks), dtype=np.float32),
                    'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
                    'mean_outer_loss': 0.}
        is_classification = self.task_type == "classification"
        if is_classification:
            results.update({'evas_before': np.zeros((num_tasks,), dtype=np.float32),
                            'evas_after': np.zeros((num_tasks,), dtype=np.float32)})
        mean_outer_loss = 0.#torch.tensor(0., device=self.device)
        if self.close_deepcopy:
            phi_t = {name: self.model.state_dict()[name] for name in self.model.state_dict()}
        else:
            phi_t = deepcopy(self.model.state_dict())

        for task_idx, (train_inputs, train_targets, val_inputs, val_targets) \
                            in enumerate(zip(*data['train'], *data['val'])):
            if self.phase == 'pretrain':
                 # [n_ways*m_tr, 1, 28, 28] -- one task
                 #print(train_inputs.shape, val_inputs.shape)
                 x, y =  torch.cat((train_inputs, val_inputs), 0), torch.cat((train_targets, val_targets), 0)
                 m = int(len(x)/2)
                 x_train, y_train, x_test, y_test = x[:m], y[:m], x[m:], y[m:]
            else:
                x_train, y_train, x_test, y_test = train_inputs, train_targets, val_inputs, val_targets

            if self.m_test == 0:
                test = False
            # Do SGD on this task
            train_per, test_per, train_loss, test_loss = self.SGD(x_train, y_train, x_test, y_test, test)
            if test:
                mean_outer_loss += test_loss
            if self.close_deepcopy:
                current_model = {name: self.model.state_dict()[name] for name in self.model.state_dict()}
            else:
                current_model = deepcopy(self.model.state_dict())
            if task_idx == 0:
                w_average = current_model
            else:
                w_average = self._interpolate_model_params(w_average, current_model, 1./(task_idx + 1))

            if not self.lazy_update:
                self.model.load_state_dict(phi_t)

            results['inner_losses'][:, task_idx] = train_loss
            results['evas_before'][task_idx] = train_per
            results['evas_after'][task_idx] = test_per
        if self.lazy_update:
            self.model.load_state_dict(phi_t)
        mean_outer_loss/num_tasks
        results['mean_outer_loss'] = mean_outer_loss#.cpu().detach().numpy()
        return w_average, results

    def meta_learner(self, batch):
        """
            dmogd
        """
        self.t = self.t + 1
        #phi_t = deepcopy(self.model.state_dict())
        if "pretrain" in self.phase:
            # x_train, y_train = batch['train']
            # x_val, y_val = batch['val']
            data = batch
            self.act_change = 0
            test = True
            task_switch = 0
        else:
            data, task_switch , env_switch, env = batch
            self.act_change = env_switch
            test = True #False
        data = tensors_to_device(data, device=self.device)
        self.m = data['train'][0].size(1)
        #print("current task sample number:", self.m)
        #============ update CL params ================
        # this works for different task numbers
        self._update_algo_params()
        if self.close_deepcopy:
            phi_t = {name: self.model.state_dict()[name] for name in self.model.state_dict()}
        else:
            phi_t = deepcopy(self.model.state_dict()) # meta params before adaptation

        #============ base learner =================
        # can learn a batch of tasks, where w_t is the average
        w_t, results = self.base_learner(data, test)
        results['eva'] = np.mean(results['evas_before'])
        results['eva_test'] = np.mean(results['evas_after'])

        if self.phase == "pretrain_test":
            # No modifications on meta-params and model-params
            return results
        elif self.phase == "continual":
            # if self.t % 10 == 1:
            #     self.show_learner_params()
            #self.update_min_max_cordinate(w_t)
            #self.update_mean_variance(w_t)
            self.env_change_detect(results)

            if self.env_change:
                self.n += 1
                self.M_n.append(1)
                #self.D_hat.append(float(self.cal_current_D_hat()))
                self.gamma = self.rho
                # self.gamma_0 = self._cal_gamma0(np.mean(self.D_hat[:-1]))
                self.update_path_length()
            elif self.t == 1:
                self.n += 1
                self.M_n.append(1)
                #self.D_hat.append(float(self.cal_current_D_hat()))
                self.gamma = self.rho#= self._cal_gamma0()

            # scale meta OGD learning rate
            else:
                self.M_n[-1] += 1
                self.gamma = self._cal_gamma0()
                #self.D_hat[self.n -1] = float(self.cal_current_D_hat())

            # ========  modulate meta learner step-size ===========
            if self.meta_algo == 'FTL':
                # Follow The Leader, here we first use a simplified version, for equal sample size
                #print("meta lr", self.meta_lr)
                self.meta_lr = 1/float(self.M_n[-1])
                self.eta_prime = np.sqrt((self.b * self.V[-1] ** 2 + self.epsilon + self.epsilon_0)/self.a)
                # calculate base learning rate
                K = np.ceil(self.K * self.m / self.inner_batch_size)
                self.base_lr = (self.eta_prime/self.kappa + 2)/( K * self.alpha)
            elif self.meta_algo == 'OGD':
                # adjust eta_0 to control the initial meta update
                # here gamma is \gamma * \kappa in paper
                self.gamma = self.gamma /np.sqrt(self.M_n[-1]) # square root schedule
                self.meta_lr = 2 * self.b * self.gamma / self.eta_prime
                # ======== adaptively update base learner step-size \eta^\prime ============
                dist = self.cal_param_square_distance(phi_t, w_t).cpu().detach().numpy()
                #print("||phi_t - w_t||^2", dist)
                # print("||phi_t||^2", self.cal_param_norm(phi_t))
                # print("||w_t||^2", self.cal_param_norm(self.model.state_dict()))
                self.eta_prime = self.eta_prime - self.gamma * \
                                    (self.a * self.kappa - (self.b * dist * self.kappa + self.epsilon + self.epsilon_0))\
                                    /(self.eta_prime ** 2)
                self.eta_prime = self.bound_eta_prime(self.eta_prime)
                # calculate base learning rate
                K = np.ceil(self.K * self.m / self.inner_batch_size)
                self.base_lr = (self.eta_prime/self.kappa + 2)/( K * self.alpha)#eta
            else:
                raise NotImplementedError
        else:
            #pretrain use constant learning rates
            #assert self.m == self.inner_batch_size
            if self.t == 1:
                print("====== in pretrain phase ====")
                print("meta lr:{}, base lr:{}".format(self.meta_lr, self.base_lr))

        # ======== update meta-model a.k.a base-model initialization/bias \phi ============
        # phi_t slow_weights
        self.model.load_state_dict(self._interpolate_model_params(phi_t, w_t, self.meta_lr))
        #print("env shift", self.act_change)
        results['task_boundary'] = task_switch
        results['env_switch'] = self.env_change
        return results


    def env_change_detect(self, results):
        if self.detect == 'static':
            # no env change detection
            self.env_change = 0
        elif self.detect == 'window':
            if (self.t) % self.win == 0:
                self.env_change = 1
            else:
                self.env_change = 0
        elif self.detect == 'oracle':
            self.env_change = self.act_change # oracle, changing point known
        elif self.detect == 'bocd':
            self.detector.update(results['mean_inner_loss'])
            flag = self.detector.detect(self.t)
            self.env_change = flag # use online bayes changing point detection
        elif self.detect == 'accuracy':
            pass
        else:
            # other env change methods, eg. threshold
            pass

    def observe(self, batch):
        """
            encounter one task per time, batch size equal to 1
        """
        self.phase = 'continual'
        results = self.meta_learner(batch)

        return results


    # ===========  learning with bound estimation  ===========
    def forward(self, task_data, test=True):
        """
            each meta iteration only encouter one task, online setting
        """
        x_train, y_train, x_test, y_test = task_data
        self.t = self.t + 1

        phi_t = {name: self.model.state_dict()[name] for name in self.model.state_dict()}
        # Do SGD on this task
        #self.show_learner_params()
        w_t, train_per, test_per, train_loss, test_loss = self.SGD(x_train, y_train, x_test, y_test, test)
        self.show_learner_params()
        #self.env_shift_measure = deepcopy(self.model.parameters())[0].view(-1).cpu().detach().numpy()
        self.update_min_max_cordinate(w_t)

        self.update_mean_variance(w_t)

        dist = self.cal_param_square_distance(phi_t, w_t).cpu().detach().numpy()
        # print(phi_t, w_t)
        print("||phi - w||", dist)
        if self.env_change:
            self.n += 1
            self.M_n.append(1)
            self.D_hat.append(self.cal_current_D_hat())
            self.meta_lr = 2 * self.b * self.hopping_lr / self.eta_0# < 1 ??
            self.model.load_state_dict({name : phi_t[name] + (w_t[name] - phi_t[name]) * self.meta_lr for name in phi_t})
            self.gamma = self.epsilon_0 * np.sqrt(((1 + self.b/self.a)*self.D_hat_guess**2 + self.epsilon/self.a)\
                /(4*self.a*self.b**2*self.D_hat_guess**2*self.epsilon_0 + self.a**2*(self.b*self.D_hat_guess**2 + self.epsilon)**2))
            self.eta_prime = self.eta_0 - self.gamma * (self.a*self.kappa \
                - (self.b*dist*self.kappa + self.epsilon + self.epsilon_0)/(self.eta_0**2))
            self.eta_prime = self.bound_eta_prime(self.eta_prime)
            self.meta_lr = 2 * self.b * self.gamma / self.eta_prime# kappa is emilinated
            self.update_path_length()
            self.dynamic_bound += 2 * np.sqrt(self.a*(self.b * self.V[-2] + self.epsilon + self.epsilon_0)) * self.M_n[-2] * self.kappa\
                + self.M_n[-2] * self.kappa * self.delta + 3/(2. * self.epsilon_0)*self.kappa \
                    * np.sqrt(self.D_hat[-2]**2 * (1 + self.b/self.a) + self.epsilon/self.a) * np.sqrt(self.M_n[-2] - 1)*\
                        np.sqrt(self.a ** 2 * (self.b* self.D_hat[-2]** 2+ self.epsilon) ** 2 + 4 * self.a * self.b**2*self.epsilon_0*self.D_hat[-2]**2)
            self.D_constant += 2 * np.sqrt(self.a*(self.b * self.V[-2] + self.epsilon + self.epsilon_0)) * self.M_n[-2] * self.kappa\
                + self.M_n[-2] * self.kappa * self.delta

        elif self.t == 1:
            print("first")
            self.n += 1
            self.M_n.append(1)
            self.D_hat.append(self.cal_current_D_hat())
            self.gamma = self.epsilon_0 * np.sqrt(((1 + self.b/self.a)*self.D_hat_guess**2 + self.epsilon/self.a)\
                /(4*self.a*self.b**2*self.D_hat_guess**2*self.epsilon_0 + self.a**2*(self.b*self.D_hat_guess**2 + self.epsilon)**2))
            self.meta_lr = 2 * self.b * self.gamma / self.eta_0
            #self.meta_lr = 2 * self.b * self.hopping_lr / self.eta_0# < 1 ??
            self.model.load_state_dict({name : phi_t[name] + (w_t[name] - phi_t[name]) * self.meta_lr for name in phi_t})
            # print("prior update", self.model.state_dict())
            self.eta_prime = self.eta_0 - self.gamma * (self.a*self.kappa \
                - (self.b*dist*self.kappa + self.epsilon + self.epsilon_0)/(self.eta_0**2))
            self.eta_prime = self.bound_eta_prime(self.eta_prime)

        # scale meta OGD learning rate
        else:
            self.M_n[-1] += 1
            self.gamma = self._cal_gamma0() /np.sqrt(self.M_n[-1]) # square root schedule
            self.meta_lr = 2 * self.b * self.gamma / self.eta_prime
            # update phi_{t+1}
            self.model.load_state_dict({name : phi_t[name] + (w_t[name] - phi_t[name]) * self.meta_lr for name in phi_t})
            self.eta_prime = self.eta_prime - self.gamma * (self.a*self.kappa \
                - (self.b*dist*self.kappa + self.epsilon + self.epsilon_0))/(self.eta_prime**2)
            self.eta_prime = self.bound_eta_prime(self.eta_prime)
            self.D_hat[self.n -1] = self.cal_current_D_hat()

        #========= cal dynamic version AER=======
        #self.dynamic_bound += 2 * np.sqrt(self.a *(self.b * self.g_V + self.epsilon + self.epsilon_0))*self.kappa + self.kappa*self.delta
        # if not self.env_change:
        DB_tmp = 2 * np.sqrt(self.a*(self.b * self.V[-1] + self.epsilon + self.epsilon_0)) * self.M_n[-1] * self.kappa\
                + self.M_n[-1] * self.kappa * self.delta + 3/(2. * self.epsilon_0)*self.kappa \
                    * np.sqrt(self.D_hat[-1]**2 * (1 + self.b/self.a) + self.epsilon/self.a) * np.sqrt(self.M_n[-1])*\
                        np.sqrt(self.a ** 2 * (self.b* self.D_hat[-1]** 2+ self.epsilon) ** 2 + 4 * self.a * self.b**2*self.epsilon_0*self.D_hat[-1]**2)
        # else:
        #     DB_tmp = 0
        tmp = [(self.b* self.D_hat[i] ** 2 + self.epsilon) ** 2 for i in range(self.n) ]
        DB = self.dynamic_bound + DB_tmp + np.sqrt((1+self.b/self.a) * np.max(self.D_hat)**2 + self.epsilon/self.a)\
                    * np.sqrt(2*self.path_length * (self.a **2/(self.epsilon_0 ** 2) * self.kappa ** 2 * np.sum(tmp)\
                         + 4 * nLA.norm(self.D_hat, 2)**2 * self.a * self.b**2 * self.kappa ** 2/self.epsilon_0))

        #========== cal static version AER========
        self.static_bound += 2 * np.sqrt(self.a *(self.b * self.g_V + self.epsilon + self.epsilon_0))*self.kappa + self.kappa*self.delta

        self.cal_global_D_hat()

        SB = self.static_bound + np.sqrt(self.t) * 3 * self.kappa / (2 * self.epsilon_0) * np.sqrt((1 + self.b/ self.a) * self.g_D_hat ** 2 + self.epsilon/self.a)\
             * np.sqrt(4 * self.a * self.b ** 2 * self.epsilon_0 * self.g_D_hat ** 2 + self.a**2 *(self.epsilon + self.b * self.g_D_hat**2) ** 2)
        #======== compare constant ===========
        hahh = 2 * np.sqrt(self.a*(self.b * self.V[-1] + self.epsilon + self.epsilon_0)) * self.M_n[-1] * self.kappa\
                + self.M_n[-1] * self.kappa * self.delta
        self.S_constant += 2 * np.sqrt(self.a *(self.b * self.g_V + self.epsilon + self.epsilon_0))*self.kappa + self.kappa*self.delta
        print("Bound estimation, time:{}, dynamic regret:{}, static regret:{}, avg test loss:{}, avg train loss:{}, act switch:{}, used switch:{}, D_cons:{}, S_cons:{}"\
            .format(self.t , DB/self.t, SB/self.t, self.ave_test_loss, self.ave_train_loss, self.act_change, self.env_change, (self.D_constant + hahh)/self.t, self.S_constant/self.t))
        #print(self.model.state_dict())
        self.show_learner_params()
        #print("meta learning rate:", self.meta_lr)
        self.estimation[self.t] = [DB/self.t, SB/self.t, self.ave_test_loss, self.ave_train_loss, self.eta_prime, self.meta_lr, self.base_lr, dist, (self.D_constant + hahh)/self.t, self.S_constant/self.t]
        self.params[self.t] = [self.D_hat[-1], self.g_D_hat, self.V[-1], self.g_V, self.M_n[-1], self.path_length]
        return train_per, test_per, train_loss, test_loss



    # ============   keep aligned with osaka benchmark pretrain phase   =======================
    def meta_train(self, dataloader, max_batches=500, verbose=True, **kwargs):
        """
            Static Env, act as batched version of Reptile
        """
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self._meta_train_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                postfix = {'outer_loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'evas_before' in results:
                    postfix['train_accuracy'] = '{0:.4f}'.format(np.mean(results['evas_before']))
                if 'evas_after' in results:
                    postfix['val_accuracy'] = '{0:.4f}'.format(np.mean(results['evas_after']))
                if 'inner_losses' in results:
                    # average inner loss
                    postfix['inner_loss'] = '{0:.4f}'.format(np.mean(results['inner_losses']))
                pbar.set_postfix(**postfix)

    def _meta_train_iter(self, dataloader, max_batches=500):
        """
            iteration results of task batches/mini-batches
        """
        n_batches = 1
        self.model.train()
        while n_batches < max_batches:
            for batch in dataloader:
                if n_batches >= max_batches:
                    break
                # train on train + val, test on train + val, test infor not useful
                self.phase = 'pretrain'
                results = self.meta_learner(batch)
                yield results
                n_batches += 1

    def meta_test(self, dataloader, max_batches=500, verbose=True, epoch=0, **kwargs):
        """
            meta test for pretrain phase
        """
        mean_outer_loss, mean_inner_loss, mean_accuracy, mean_accuracy_before, count = 0., 0., 0., 0., 0
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'evas_before' in results:
                    mean_accuracy_before += (np.mean(results['evas_before'])
                        - mean_accuracy_before) / count
                    postfix['test_train_accuracy'] = '{0:.4f}'.format(mean_accuracy_before)
                if 'evas_after' in results:
                    mean_accuracy += (np.mean(results['evas_after'])
                        - mean_accuracy) / count
                    postfix['test_val_accuracy'] = '{0:.4f}'.format(mean_accuracy)
                if 'inner_losses' in results:
                    mean_inner_loss += (np.mean(results['inner_losses'])
                        - mean_inner_loss) / count
                    postfix['inner_loss'] = '{0:.4f}'.format(mean_inner_loss)
                pbar.set_postfix(**postfix)

        results = {
            'mean_outer_loss': mean_outer_loss,
            'evas_before': mean_accuracy_before,
            'evas_after': mean_accuracy,
            'mean_inner_loss': mean_inner_loss,
        }

        return results

    def evaluate_iter(self, dataloader, max_batches=500):
        """
           iteration results of task batches/mini-batches
        """
        n_batches = 0
        self.model.eval()
        while n_batches < max_batches:
            for batch in dataloader:
                if n_batches >= max_batches:
                    break
                batch = tensors_to_device(batch, device=self.device)
                # same as continual phase, train on train data, test on validation data
                self.phase = 'pretrain_test'
                results = self.meta_learner(batch)
                yield results
                n_batches += 1
