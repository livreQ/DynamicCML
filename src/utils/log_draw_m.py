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

folder_prefix = "./mnist_acc/acc_m_seed"



for method in methods:
    plot_all = []
    m_all = []
    for i in range(3):
        ms = []
        test_acc = []
        result = OrderedDict()
        f_all = load_files(folder_prefix + str(i))
        f_method = filter(lambda x: method_filter(x, method), f_all)
        for f_name in f_method:
            print(f_name)
            m = int(f_name.split('.')[5])
            # hazards.append(hazard)
            lines = read_log(folder_prefix + str(i) + "/" + f_name)
            #print(lines)
            acc = filter(lambda x: content_filter(x, "Test acc"), lines)
            tmp = []
            for line in acc:
                #print(line)
                tmp.append(float(line.split(',')[2].split(':')[1]))
                if int(line.split(',')[0].split(':')[1]) == 200:
                    result[m] = np.average(tmp)
                    break
        for m, acc in sorted(result.items(),key=lambda x:x[0], reverse=False):
            ms.append(m)
            test_acc.append(acc)
        m_all.append(ms[:7])
        plot_all.append(test_acc[:7])
    print(plot_all)
    acc_mean = np.mean(plot_all, 0)
    acc_var = np.var(plot_all,0)
    print(acc_var)
    #plt.plot(hazards_all[0], acc_mean,'-', label=method)
    plt.errorbar(m_all[0], acc_mean, yerr=acc_var, fmt='-^', label=method)
    print(m_all)
    print(test_acc)
    plt.legend(loc='lower right',ncol=2)
plt.xlabel('sample number $m$')
plt.ylabel('Average test acc at t=200')
# plt.title("Rotated MNIST: Average Test accuracy with different env changing randomness")
plt.savefig('sample_number_mnist.pdf', format='pdf', bbox_inches='tight')
plt.show()


