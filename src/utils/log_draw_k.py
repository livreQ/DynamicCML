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

methods = ["oracle", "detect", "window", "static"]
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

folder_prefix = "./mnist_acc/acc_k_seed"



for method in methods:
    plot_all = []
    ks_all = []
    for i in range(1):
        ks = []
        test_acc = []
        result = OrderedDict()
        f_all = load_files(folder_prefix + str(i))
        f_method = filter(lambda x: method_filter(x, method), f_all)
        for f_name in f_method:
            print(f_name)
            k = int(f_name.split('.')[4])
            # ks.append(k)
            lines = read_log(folder_prefix + str(i) + "/" + f_name)
            #print(lines)
            acc = filter(lambda x: content_filter(x, "Test acc"), lines)
            tmp = []
            for line in acc:
                #print(line)
                tmp.append(float(line.split(',')[2].split(':')[1]))
                if int(line.split(',')[0].split(':')[1]) == 200:
                    result[k] = np.average(tmp)
                    break
        for k, acc in sorted(result.items(),key=lambda x:x[0], reverse=False):
            ks.append(k)
            test_acc.append(acc)
        ks_all.append(ks[:4])
        plot_all.append(test_acc[:4])
    print(plot_all)
    acc_mean = np.mean(plot_all, 0)
    acc_var = np.var(plot_all,0)
    print(acc_var)
    #plt.plot(ks_all[0], acc_mean,'-', label=method)
    plt.errorbar(ks_all[0], acc_mean, yerr=acc_var, fmt='-^', label=method)
    print(ks)
    print(test_acc)
    plt.legend(loc='lower right',ncol=2)
plt.xlabel('inner epoch K')
plt.ylabel('Average test acc at t=200')
# plt.title("Rotated MNIST: Average Test accuracy with different env changing randomness")
plt.savefig('k_mnist.pdf', format='pdf', bbox_inches='tight')
plt.show()


