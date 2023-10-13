#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Fri May  6 15:51:41 2022
# File Name: run_guassian.py
# Description:
"""

from __future__ import absolute_import,division, print_function
from xmlrpc.client import boolean
import  torch, os
import  numpy as np
import  argparse
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from src.learners.dmogd import DMOGD
from src.dataloader.toy_datasets import Switching2DGaussianDataset
from copy import deepcopy
from src.envdetector.env_change_detection import OnlineEnvChangePointDetection

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
    prefix = "hazard:" + str(args.hazard) + ",sample_num:" + str(args.sample_num) + ",method:" + method[args.method - 1]\
        + ",inner K:" + str(args.K) + ",win:" + str(args.win)
    
    if args.hazard <= 0:
        args.hazard = np.float_power(10, args.hazard)
    print(args)
    config = [
        ('pd', [2, 2]),# 2-dim distance function
    ]

    dmogd = DMOGD(args, config).to(device)
    tmp = filter(lambda x: x.requires_grad, dmogd.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))

    print(dmogd)
    dmogd.show_learner_params()
   
    eocd = init_env_change(args)
    print('Total trainable tensors:', num)
    data = Switching2DGaussianDataset(args, False)

    #online setting, we can only see a task once
    fig = plt.figure()
    ax = plt.subplot(111, aspect='equal')
    ax.set(xlim=(-10, 10), ylim=(-10, 10))
    hind_env_mean, pred_env_mean, actual_env_mean = [], [], []
    predict = []
    ac_switch = []
    pred_change = []
    for t in range(args.horizon):
        #print(t)
        x_train, y_train, x_test, y_test, env_mean, switch_flag = data[t]
        x_train = torch.from_numpy(x_train).to(device)
        #print(x_train)
        y_train = torch.from_numpy(y_train).to(device)
        x_test = torch.from_numpy(x_test).to(device)
        y_test = torch.from_numpy(y_test).to(device)
        dmogd.act_change = switch_flag
        
        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        train_per, test_per, train_loss, test_loss  = dmogd.forward((x_train, y_train, x_test, y_test), True)# w_t
        env_mean_estimation = deepcopy(dmogd.net.parameters()[0].cpu().detach().numpy())
        predict.append(train_per)
        # plot prior:
        #plt.plot(env_mean_estimation[0], env_mean_estimation[1], 'o', label='predicted env mean at {}'.format(t))

        # plot task data points:
        # plt.plot(x_train[:, 0], x_train[:, 1], '.', color='gray')
        # plt.plot(x_test[:, 0], x_test[:, 1], '.', color='red')
  
        # plot posterior:
        #plt.plot(train_per[0], train_per[1], '.')
        flag = env_change_detect(eocd, train_per, t)
        if flag:
            pred_change.append(t)

        if switch_flag or t==0:
            # ell = Ellipse(xy=(env_mean_estimation[0], env_mean_estimation[1]),
            #       width=0.1, height=0.1,
            #       angle=0, color='blue')
            # ell.set_facecolor('none')
            # ax.add_artist(ell)
            ac_switch.append(t)
            hind_env_mean.append(deepcopy(dmogd.mean_[0].cpu().detach().numpy()))#estimation of avg mean 
            actual_env_mean.append(env_mean)
            pred_env_mean.append(env_mean_estimation)
            
            # plt.plot(env_mean[0], env_mean[1], 'o', label='actual env mean at {}'.format(t))
            # plt.plot(dmogd.mean_[0].cpu().detach().numpy()[0], dmogd.mean_[0].cpu().detach().numpy()[1], '.', label='hindsight mean at {}'.format(t))
            # plt.plot(env_mean_estimation[0], env_mean_estimation[1], 'x', label='predicted env mean at {}'.format(t))
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
        print("Task:{},Train acc:{}, Test acc:{}, actucal change:{}, detect change:{}, switching method:{}, check:{}"\
            .format(t+1, train_per, test_per, switch_flag, flag, args.method, dmogd.env_change))
        # if t == 0:
        #     break

    # plt.plot(np.array(hind_env_mean)[:,0], np.array(hind_env_mean)[:,1], '->', label='hind env mean', color='black')
    # plt.plot(np.array(actual_env_mean)[:,0], np.array(actual_env_mean)[:,1],'-*', label='actual env mean', color='black')
    # plt.plot(np.array(pred_env_mean)[:,0], np.array(pred_env_mean)[:,1], '-x', label='predicted env mean', color='black')
    plt.plot(np.asarray(predict)[:,0], np.asarray(predict)[:,1], '.', label="task mean estimations")
    plt.plot(np.array(hind_env_mean)[:,0], np.array(hind_env_mean)[:,1], '->', label='hindsight env mean')
    plt.plot(np.array(actual_env_mean)[:,0], np.array(actual_env_mean)[:,1],'-*', label='actual env mean')
    plt.plot(np.array(pred_env_mean)[:,0], np.array(pred_env_mean)[:,1], '-x', label='predicted env mean')
    plt.legend(loc='lower right',ncol=2)

    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title("2D Gaussian mean estimation in a dynamic environment")
    plt.savefig(prefix + '.pdf', format='pdf', bbox_inches='tight')
    # plt.show()
    print(np.sum(ac_switch))
    eocd[0].plot_posterior(np.asarray(predict)[:,0], ac_switch, pred_change)
    eocd[1].plot_posterior(np.asarray(predict)[:,1], ac_switch, pred_change) 
    with open("./result/" + prefix + '.npy', 'wb') as f:
        np.save(f, hind_env_mean, )
        np.save(f, actual_env_mean)
        np.save(f, pred_env_mean)
        np.save(f, predict)
        np.save(f, ac_switch)
        np.save(f, pred_change)
        np.save(f, dmogd.estimation)
        np.save(f, dmogd.params)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    #data settings
    argparser.add_argument('--horizon', type=int, help='horizon of tasks', default=200)
    argparser.add_argument('--paradim', type=int, help='model parameter dimension', default=2)
    argparser.add_argument('--hazard', type=float, help='hazard', default=0.05)
    argparser.add_argument('--sample_num', type=int, help='sample number per task', default=100)
    argparser.add_argument('--test_num', type=int, help='test sample number per task', default=40)
    argparser.add_argument('--sigma_eps', type=int, help='noise added on sinusoid', default=0.001)
    # loss function settings
    argparser.add_argument('--loss', type=str, help='specify loss function', default='mse')
    argparser.add_argument('--task', type=str, help='task type, classification, regression, mean_estimatin', default='mean_estimation')
    # base learner settings
    argparser.add_argument('--K', type=int, help='inner SGD steps', default=5)
    argparser.add_argument('--K_test', type=int, help='inner test SGD steps', default=5)
    # meta learner settings
    argparser.add_argument('--L', type=int, help='Lipschitz constant', default=30)
    argparser.add_argument('--alpha', type=int, help='alpha quadratic growth', default=2)
    argparser.add_argument('--D_hat_guess', type=int, help='D_hat initialization', default=5)
    argparser.add_argument('--rho', type=float, help='hopping learning rate', default=0.8)
    # output setting
    argparser.add_argument('--eva_bound', type=bool, help='evaluate bound or not', default=1)
    argparser.add_argument('--method', type=int, help='switch time methods, 1-oracle known switch time;\
         2-detected switch time; 3-no switch info, treate as static, 4-slideing window', default=1)
    argparser.add_argument('--win', type=int, help='spliting window size', default=10)
    args = argparser.parse_args()
    print(args)
    start_learn(args)
    