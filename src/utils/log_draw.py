#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Tue May 10 22:48:42 2022
# File Name: src/utils/log_draw.py
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
from src.utils.datasets import Switching2DGaussianDataset
from copy import deepcopy
from src.utils.env_change_detection import OnlineEnvChangePointDetection
from collections import OrderedDict
draw_bound = True

methods = ["oracle", "detect", "window", "static", 'window-1']
def load_files(path):    
    files = os.listdir(path)     
    return files

def method_filter(f_name, method):

    if method in f_name:
        return True
    else:
        return False
    
def read_log(path):
    with open(path, 'r') as f:
        return f.readlines()    

def content_filter(lines, content):
    if content in lines:
        return True
    else:
        return False

folder_prefix = "./mnist_acc/acc_seed"

acc_win = True

for method in methods:
    plot_all = []
    hazards_all = []
    for i in range(1):
        hazards = []
        test_acc = []
        result = OrderedDict()
        f_all = load_files(folder_prefix + str(i))
        f_method = filter(lambda x: method_filter(x, method), f_all)     
        for f_name in f_method:
            print(f_name)
            hazard = int(f_name.split('.')[3])/100.
            # hazards.append(hazard)
            lines = read_log(folder_prefix + str(i) + "/" + f_name)
            #print(lines)
            acc = filter(lambda x: content_filter(x, "Test acc"), lines)
            tmp = []
            for line in acc:
                #print(line)
                tmp.append(float(line.split(',')[2].split(':')[1]))
                if int(line.split(',')[0].split(':')[1]) == 200:
                    result[hazard] = np.average(tmp)
                    break
        for hazard, acc in sorted(result.items(),key=lambda x:x[0], reverse=False):
            hazards.append(hazard)
            test_acc.append(acc)
        hazards_all.append(hazards[:7])
        plot_all.append(test_acc[:7])
    if acc_win and method == "window-1":
        print("=============")
        hazards = []
        test_acc = []
        result = OrderedDict()
        f_acc_win1 = load_files("./mnist_acc/acc_win")
        f_method_win = filter(lambda x: method_filter(x, "window"), f_acc_win1)
        for f_name in f_method_win:
            print(f_name)
            hazard = int(f_name.split('.')[3])/100.
            # hazards.append(hazard)
            lines = read_log("./mnist_acc/acc_win/" + f_name)
            #print(lines)
            acc = filter(lambda x: content_filter(x, "Test acc"), lines)
            tmp = []
            for line in acc:
                #print(line)
                tmp.append(float(line.split(',')[2].split(':')[1]))
                if int(line.split(',')[0].split(':')[1]) == 200:
                    result[hazard] = np.average(tmp)
                    break
        for hazard, acc in sorted(result.items(),key=lambda x:x[0], reverse=False):
            print(acc)
            hazards.append(hazard)
            test_acc.append(acc)
        print(hazards, test_acc)
        hazards_all.append(hazards[:7])
        plot_all.append(test_acc[:7])
    
    print(plot_all)
    acc_mean = np.mean(plot_all, 0)
    acc_var = np.std(plot_all,0)
    print(acc_var)
    #plt.plot(hazards_all[0], acc_mean,'-', label=method)
    plt.errorbar(hazards_all[0], acc_mean, yerr=acc_var, fmt='-^', label=method)
    print(hazards)
    print(test_acc)
    plt.legend(loc='lower right',ncol=2)
plt.xlabel('hazard')
plt.ylabel('Average test acc at t=200')
# plt.title("Rotated MNIST: Average Test accuracy with different env changing randomness")
plt.savefig('hazard_mnist.pdf', format='pdf', bbox_inches='tight')
plt.show()


