#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Tue May 10 00:16:22 2022
# File Name: src/utils/draw.py
# Description:
"""
from __future__ import absolute_import,division, print_function
from cProfile import label
from tkinter import font
from xmlrpc.client import boolean
import  torch, os
import  numpy as np
import matplotlib as mpl
import  argparse
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from src.learners.dmogd import DMOGD
from src.utils.datasets import Switching2DGaussianDataset
from copy import deepcopy
from src.utils.env_change_detection import OnlineEnvChangePointDetection
from collections import OrderedDict
draw_bound = False
def load_files(path):    
    files = os.listdir(path)     
    return files

def hazard_change(f_name):
    if "sample_num:100" in f_name and "innerK:10" in f_name:
        return True
    else:
        return False

def K_change(f_name):
    if "sample_num:40" in f_name and "hazard:0.05" in f_name:
        return True
    else:
        return False

def m_change(f_name):
    if "inner K:25" in f_name and "hazard:0.05" in f_name:
        return True
    else:
        return False

f_all = load_files("./result/")

def draw_hazard(f_all):

    results_change_with_hazard = filter(hazard_change, f_all)

    hazard_change_data = {"oracle":OrderedDict(),"detect":OrderedDict(),"window":OrderedDict(),"static":OrderedDict()}
    hazards = []
    test = []
    for f_name in results_change_with_hazard:
        print(f_name)
        tmp = f_name.split(',')
        #hazard = np.float_power(10, float(tmp[0].split(':')[1]))
        hazard = float(tmp[0].split(':')[1])
        method = tmp[2].split(':')[1]
        with open("./result/" + f_name, 'rb') as f:
            print(hazard)
            hind_env_mean =  np.load(f, allow_pickle=True)
            actual_env_mean = np.load(f, allow_pickle=True)
            pred_env_mean = np.load(f, allow_pickle=True)
            predict = np.load(f, allow_pickle=True)
            ac_switch= np.load(f, allow_pickle=True)
            pred_change  = np.load(f, allow_pickle=True)
            estimation = np.load(f, allow_pickle=True)
            estimation = estimation.item()
            hazard_change_data[method][hazard] =[estimation[200][0], estimation[200][1], estimation[200][2]]
            hazards.append(hazard)

    fig = plt.figure()
    # ax = plt.subplot(111, aspect='equal')
    for method in hazard_change_data:
        h , d = [], []
        for hazard, data in sorted(hazard_change_data[method].items(),key=lambda x:x[0], reverse=False):
            if hazard <= 0:
                hazard = np.float_power(10, hazard)
            h.append(hazard)
            d.append(np.log(data[2]))
            print(h[1:])
            print(d[1:])
            # if method != 'static':
        plt.plot(h[1:], d[1:],'-', label=method)
        # plt.plot(hazard_change_data["detect"][:,0], hazard_change_data["detect"][:,3],'-.', label='obcd')
        # plt.plot(hazard_change_data["window"][:,0], hazard_change_data["window"][:,3],'-*', label='window')
        # plt.plot(hazard_change_data["static"][:,0], hazard_change_data["static"][:,3],'--', label='no slot info')
        #ax.set_xscale('log')
    # plt.legend(loc='lower right',ncol=2)
    plt.xlabel('$p$')
    plt.ylabel('Average test loss (log scale) at t=200')
    # plt.title("Performance of slot methods w.r.t different env changing randomness")
    plt.legend(loc='center right', ncol=2)
    plt.savefig('hazard.pdf', format='pdf', bbox_inches='tight')
    plt.show()
       
f_font = 12

def draw_bound(f_all, bound):
    results_change_with_hazard = filter(hazard_change, f_all)
    for f_name in results_change_with_hazard:
        print(f_name)
        with open("./result/" + f_name, 'rb') as f:
            hind_env_mean =  np.load(f, allow_pickle=True)
            actual_env_mean = np.load(f, allow_pickle=True)
            pred_env_mean = np.load(f, allow_pickle=True)
            predict = np.load(f, allow_pickle=True)
            ac_switch= np.load(f, allow_pickle=True)
            pred_change  = np.load(f, allow_pickle=True)
            estimation = np.load(f, allow_pickle=True)
            params = np.load(f, allow_pickle=True)

        if not bound:
            fig = plt.figure()
            ax = plt.subplot(111, aspect='equal')
            ax.set(xlim=(-10, 10), ylim=(-10, 10))
            plt.plot(np.asarray(predict)[:,0], np.asarray(predict)[:,1], '.', label="task mean estimations")
            plt.plot(np.array(hind_env_mean)[:,0], np.array(hind_env_mean)[:,1], '->', label='hindsight env mean')
            plt.plot(np.array(actual_env_mean)[:,0], np.array(actual_env_mean)[:,1],'-*', label='actual env mean')
            plt.plot(np.array(pred_env_mean)[:,0], np.array(pred_env_mean)[:,1], '-x', label='predicted env mean')
            plt.legend(loc='lower right',ncol=2)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title("2D Gaussian mean estimation in a dynamic environment")
            plt.savefig('gaussian.pdf', format='pdf', bbox_inches='tight')
            plt.show()
        else:
            # self.estimation[self.t] = [DB/self.t, self.static_bound/self.t, self.ave_test_loss, self.eta_prime, self.meta_lr, self.base_lr, dist]
            plt.rcParams['xtick.labelsize'] = 10
            plt.rcParams['ytick.labelsize'] = 10
            plt.rc('xtick', labelsize=8) 
            plt.rc('ytick', labelsize=8)
            plt.rc('font',family='Times New Roman')
            plt.rcParams['mathtext.fontset']='cm'
            plt.rc('legend',fontsize=10) 
            plt.rc('axes', titlesize=10)     # fontsize of the axes title
            plt.rc('axes', labelsize=10) 
            # mm = 1/25.4

            fig, axes = plt.subplots(4, 1, figsize=(8,7))
            print(type(estimation))
            print(np.shape(axes))
            ax1, ax2, ax3, ax4= axes

            # ax3 = ax2.twinx()
            # ax5 = ax4.twinx()

            r_dy = []
            r_st = []
            test_loss = []
            train_loss = []
            D_cons = []
            S_cons = []
            meta_lr = []
            base_lr = []
            distance = []
            estimation = estimation.item()
            params = params.item()
            D_hat = []
            g_D_hat = []
            V = []
            g_V = []
            path_length = []
            inds = range(1,125)
            for t in inds:#estimation:
                #print(estimation[t])
                r_dy.append(estimation[t][0])
                r_st.append(estimation[t][1])
                test_loss.append(estimation[t][2])
                train_loss.append(estimation[t][3])
                meta_lr.append(estimation[t][5])
                base_lr.append(estimation[t][6])
                distance.append(estimation[t][7])
                D_cons.append(estimation[t][-2])
                S_cons.append(estimation[t][-1])
                D_hat.append(params[t][0])
                g_D_hat.append(params[t][1])
                V.append(params[t][2])
                g_V.append(params[t][3])
                path_length.append(params[t][5])
                # meta_lr.append(estimation[t][4])
                # base_lr.append(estimation[t][5])

            ac_switch =  filter(lambda x: x <= max(inds), ac_switch)
            pred_change = filter(lambda x: x <= max(inds), pred_change)
            # ax1.plot(np.log(test_loss), label='test loss', linewidth=1)
            # # ax1.plot(train_loss, label='train loss')
            # ax2.plot(np.log(r_dy), label='dynamic AER', linewidth=1)
            # ax2.plot(np.log(r_st), label='static AER', linewidth=1)
            ax3.plot(meta_lr, label='meta_lr', linewidth=1)
            ax4.plot(base_lr, label='base_lr', linewidth=1)
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])
            ax3.set_yticklabels([])
            ax4.set_yticklabels([])
            # make a plot with different y-axis using second axis object
            # ax6.plot(distance, label="$\phi - w_t$")
            # ax4.plot(D_hat, label='D_n')
            # ax4.plot(g_D_hat, label='D')
            # ax5.plot(V, label='V_n')
            # ax5.plot(g_V, label='V')
            # ax6.plot(path_length, label='path length')
            # ax2.plot(D_cons, label='dynamic cons')
            # ax2.plot(S_cons, label='static cons')
            for cp in ac_switch:
                ax1.axvline(cp, c='red', ls='dotted', linewidth=1)
                ax2.axvline(cp, c='red', ls='dotted', linewidth=1)
                ax3.axvline(cp, c='red', ls='dotted', linewidth=1)
                # ax4.axvline(cp, c='red', ls='dotted')
                # ax5.axvline(cp, c='red', ls='dotted')
            
            for cp in pred_change:
                # ax1.axvline(cp, c='green', ls='dotted')
                ax4.axvline(cp, c='green', ls='dotted', linewidth=1)
            ax1.set_ylabel('log avg test loss')
            # ax1.set_title('test loss change with task horizon')
            ax2.set_ylabel('log AER')
            ax2.legend(loc='center right', ncol=1)
            #ax3.set_ylabel('Static AER')
            ax3.set_ylabel('meta lr')
            ax4.set_ylabel('base lr')
            # ax6.set_ylabel('$\phi - w_t$')
            # ax2.set_title('AER change with task horizon')
            ax1.set_xlabel('t')
            ax2.set_xlabel('t')
            ax3.set_xlabel('t')
            ax4.set_xlabel('t')
            # ax5.set_xlabel('t')
            # ax6.set_xlabel('t')
            
            #ax1.title("2D Gaussian mean estimation in a dynamic environment")
            plt.savefig('gaussian.pdf', format='pdf', bbox_inches='tight')
            plt.show()


if __name__ == '__main__':
    draw_bound(f_all, bound=True)
    #draw_hazard(f_all)