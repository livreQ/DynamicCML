#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Fri May  6 15:52:33 2022
# File Name: run_rotatedMNIST.py
# Description:
"""


from __future__ import absolute_import,division, print_function
from math import fabs
from xmlrpc.client import boolean
import  torch, os
import  numpy as np
import  argparse
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from src.learners.dmogd import DMOGD
from src.dataloader.toy_datasets import Switching2DGaussianDataset, SwitchingRotatedMNIST
from copy import deepcopy
from src.envdetector.env_change_detection import OnlineEnvChangePointDetection
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(168)
torch.cuda.manual_seed_all(168)
np.random.seed(168)

method = ['oracle', 'detect', 'window', 'static']

def init_env_change(args):
    eocd = []
    for i in range(args.paradim):
        eocd.append(OnlineEnvChangePointDetection(args))
    return eocd

def env_change_detect(eocd, w_t, t):
    test_flag = 0
    #max_ind = np.zeros(eocd[0].horizon + 1)
    R = np.zeros((eocd[0].horizon + 1, eocd[0].horizon + 1))
    for i in range(eocd[0].w_dim):
        R += eocd[i].detect(w_t[i])
        #max_ind += eocd[i].decide_change_point(t + 1)
    max_ind = R.argmax(axis=1)
    if max_ind[t+1] < max_ind[t]:# and max[t] > 0.5:
        test_flag =1
    else:
        test_flag = 0
        
    return test_flag

def start_learn(args):
    print(args)
    prefix = "hazard:" + str(args.hazard) + ",sample_num:" + str(args.sample_num) + ",method:" + method[args.method - 1]\
        + ",inner K:" + str(args.K) + ",win:" + str(args.win)
    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('flatten', []),
        ('linear', [10, 64])
    ]

    dmogd = DMOGD(args, config).to(device)
    tmp = filter(lambda x: x.requires_grad, dmogd.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    
    print(dmogd)
    dmogd.show_learner_params()
   
    eocd = init_env_change(args)
    print('Total trainable tensors:', num)
    data = SwitchingRotatedMNIST(args, train=False)
    # data_loader = DataLoader(
    #     SwitchingRotatedMNIST(args, train=False),
    #     batch_size=1,
    #     shuffle=False)
    #online setting, we can only see a task once
    # fig = plt.figure()
    # ax = plt.subplot(111, aspect='equal')
    # ax.set(xlim=(-10, 10), ylim=(-10, 10))
    # hind_env_mean, pred_env_mean, actual_env_mean = [], [], []
    predict = []
    ac_switch = []
    pred_change = []
    train_acc = []
    test_acc = []
    # for t, (x_train, y_train, x_test, y_test, switch_flag, rot) in enumerate(data_loader):
    for  t in range(dmogd.task_horizon):
        #print(t)
        x_train, y_train, x_test, y_test, switch_flag, rot = data[t]
        x_train = x_train.to(device)
        x_test = x_test.to(device)
        y_train = torch.from_numpy(y_train).to(device)
        y_test = torch.from_numpy(y_test).to(device)
        dmogd.act_change = switch_flag
        #dmogd.env_change = switch_flag
        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        train_per, test_per, train_loss, test_loss  = dmogd.forward((x_train, y_train, x_test, y_test), True)# w_t
        env_mean_estimation = deepcopy(dmogd.net.parameters()[0].cpu().detach().numpy())
        train_acc.append(train_per)
        test_acc.append(test_acc)
        print(dmogd.w_t)
        predict.append(dmogd.w_t)
        # plot prior:
        #plt.plot(env_mean_estimation[0], env_mean_estimation[1], 'o', label='predicted env mean at {}'.format(t))

        # plot task data points:
        #plt.plot(x_train[:, 0], x_train[:, 1], '.', color='gray')
        # plot posterior:
        # plt.plot(train_per[0], train_per[1], '.')
        flag = env_change_detect(eocd, dmogd.w_t, t)
        if flag:
            pred_change.append(t)

        if switch_flag or t==0:
            ac_switch.append(t)

        if args.method == 1:
            dmogd.env_change = switch_flag # oracle, changing point known
        elif args.method == 2:
            dmogd.env_change = flag # use online bayes changing point detection
        elif args.method == 3: # spliting window 
            if (t+1) % args.win == 0:
                dmogd.env_change = 1
            else:
                dmogd.env_change = 0
        else:
            pass
            
        print("Task:{},Train loss:{:.4}".format(t+1, np.mean(train_loss)))
        print("Task:{},Test loss:{:.4}".format(t+1, np.mean(test_loss)))
        print("Task:{},Train acc:{}, Test acc:{}, actucal change:{}, detect change:{}, switching method:{}"\
            .format(t+1, train_per, test_per, switch_flag, flag, args.method))

    print(np.sum(ac_switch))
    # eocd[0].plot_posterior(np.asarray(predict)[:,0], ac_switch, pred_change)
    # eocd[1].plot_posterior(np.asarray(predict)[:,1], ac_switch, pred_change) 
    with open("./result/" + prefix + '.npy', 'wb') as f:
        #np.save(f, predict)
        np.save(f, train_acc)
        np.save(f, test_acc)
        np.save(f, ac_switch)
        np.save(f, pred_change)
        np.save(f, dmogd.estimation)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    #data settings
    argparser.add_argument('--horizon', type=int, help='horizon of tasks', default=100)
    argparser.add_argument('--paradim', type=int, help='model parameter dimension', default=10)
    argparser.add_argument('--hazard', type=float, help='hazard', default=0.05)
    argparser.add_argument('--sample_num', type=int, help='sample number per task', default=100)
    argparser.add_argument('--test_num', type=int, help='test sample number per task', default=40)
    # loss function settings
    argparser.add_argument('--loss', type=str, help='specify loss function', default='cross_entropy')
    argparser.add_argument('--task', type=str, help='task type, classification, regression, mean_estimatin', default='classification')
    # base learner settings
    argparser.add_argument('--K', type=int, help='inner SGD steps', default=40)
    argparser.add_argument('--K_test', type=int, help='inner test SGD steps', default=20)
    # meta learner settings
    argparser.add_argument('--L', type=int, help='Lipschitz constant', default=100)
    argparser.add_argument('--alpha', type=int, help='alpha quadratic growth', default=4)
    argparser.add_argument('--D_hat_guess', type=int, help='D_hat initialization', default=100000)
    argparser.add_argument('--rho', type=float, help='hopping learning rate', default=0.6)
    # output setting
    argparser.add_argument('--eva_bound', type=bool, help='evaluate bound or not', default=True)
    argparser.add_argument('--method', type=int, help='switch time methods, 1-oracle known switch time;\
         2-detected switch time; 3-slideing window; 4-static', default=1)
    argparser.add_argument('--win', type=int, help='spliting window size', default=10)
    argparser.add_argument('--root', type=str, help='rotated mnist data path', default="./data/")
    args = argparser.parse_args()
    start_learn(args)
    