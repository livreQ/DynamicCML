#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Thu May  5 11:48:08 2022
# File Name: datasets.py
# Description:
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
# from skimage import io
from src.utils.utils import listdir_nohidden
import torchvision
import  argparse

import os.path as osp
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from torchvision import transforms
from torchvision import datasets, transforms
from torchvision.utils import save_image

torch.manual_seed(168)
torch.cuda.manual_seed_all(168)
np.random.seed(168)

def uniform_sample(x):
    x_lower, x_upper = x
    return x_lower + np.random.rand()*(x_upper-x_lower)


def truncated_resample(x, size=100, range_=[(-5, 5),(-5, 5)], mean_=None, var_=None):
    for i in range(size):
        #print(mean_, var_)
        #print(x[i][0], x[i][1])
        while (x[i][0] < range_[0][0] or x[i][0]> range_[0][1]) or (x[i][1] < range_[1][0] or x[i][1]> range_[1][1]):
            x[i] = np.random.multivariate_normal(mean=mean_, cov=var_,size=1).astype(np.float32)[0]
            #print(x[i])
    return x

def gen_range(xl, xr, n):
    assert xr > xl
    range_list = []
    for i in range(n):
        interval = (xr-xl)/n
        range_list.append([xl, xl + interval])
        xl = xl + interval
    return range_list

class Switching2DGaussianDataset(Dataset):
    def __init__(self, config, train=True):
        self.env_mean = [-6, -6]
        self.change_rate = 1
        self.env_var = np.diag([1, 1])
        self.task_var = np.diag([0.1, 0.1])
        self.x_range = [-10., 10.]
        self.y_range = [-10., 10.]
        self.horizon = config.horizon# T
        self.n_env = 0
        self.switch_prob = config.hazard
        self.m_t = config.sample_num # train sample number for each task, unsupervised
        self.m_test = config.test_num # train sample number for each task, unsupervised
        self.train = train
        self.sample()
    def __len__(self):
        # task numbers
        return self.horizon
            
    def sample(self, return_lists=True):
        x_dim = 2
        y_dim = 2
        
        self.x_train = np.zeros((self.horizon, self.m_t, x_dim))
        self.y_train = np.zeros((self.horizon, self.m_t, y_dim))
        self.x_test = np.zeros((self.horizon, self.m_test, x_dim))
        self.y_test = np.zeros((self.horizon, self.m_test, y_dim))
        self.env_mu_list = []
        self.switch_flag = []
        for t in range(self.horizon):
            #print(t)
            if np.random.rand() < self.switch_prob or t==0:
                # random sample env idx for amp, phase and freq
                #print("env change")
                self.env_mean[0] += self.change_rate
                self.env_mean[1] += self.change_rate
                if t > 0:
                    self.switch_flag.append(1)
                self.n_env += 1
            self.switch_flag.append(0)
            self.env_mu_list.append([self.env_mean[0], self.env_mean[1]])
            #print(self.env_mean)
            # sample task from env
            mu = np.random.multivariate_normal(mean=self.env_mean, cov=self.env_var, size=1)
            # print("task_mean", mu)
            z = np.random.multivariate_normal(mean=mu[0], cov=self.task_var,
                size=self.m_t).astype(np.float32)
            #z = truncated_resample(z, size=self.m_t, range_=[self.x_range, self.y_range], mean_=mu[0], var_=self.task_var)
            self.x_train[t,:] = z
            z_test = np.random.multivariate_normal(mean=mu[0], cov=self.task_var,
                size=self.m_test).astype(np.float32)
            #z_test = truncated_resample(z_test, size=self.m_test, range_=[self.x_range, self.y_range], mean_=mu[0], var_=self.task_var)
            # print(np.shape(z_test)[0])
            # inds = np.arange(np.shape(z_test)[0])
            # print(inds,z_test[inds])
            # np.random.shuffle(inds)
            self.x_test[t,:] = z_test#[inds]
            print(np.mean(z, 0), np.mean(z_test, 0))
    
    def __getitem__(self,idx,train=True):
  
        #train_x, train_y, test_x, test_y, env_mu_list, switch_flag = self.sample()
        if self.train:
            return self.x_train[idx], self.y_train[idx], self.env_mu_list[idx], self.switch_flag[idx]
        else:
            return self.x_train[idx], self.y_train[idx], self.x_test[idx], self.y_test[idx], self.env_mu_list[idx], self.switch_flag[idx]            


class SwitchingSinusoidDataset:
    def __init__(self, config, data_length=100000000):
        self.env_ = [4,4,4]
        self.amp_range = gen_range(0.1, 5.0, self.env_[0])
        self.phase_range = gen_range(0, 3.14, self.env_[1])
        self.freq_range = gen_range(0.999, 1.0, self.env_[2])
        self.x_range = [-5., 5.]
        self.horizon = config.horizon# T
        self.n_env = np.product(self.env_)# N
        self.switch_prob = config.hazard
        self.m_t = config.sample_num # sample number for each task 
        self.data_length = data_length
        self.sigma_eps = eval(config.sigma_eps)[0]
        self.noise_std = np.sqrt(self.sigma_eps)
        
    def __len__(self):
        return self.data_length-self.horizon
            
    def sample(self, return_lists=True):
        x_dim = 1
        y_dim = 1
        
        x = np.zeros((self.horizon, self.m_t, x_dim))
        y = np.zeros((self.horizon, self.m_t, y_dim))

        amp_range_list = []
        phase_range_list = []
        freq_range_list = []
        switch_times = []
        for i in range(self.horizon):
            if np.random.rand() < self.switch_prob or i==0:
                # random sample env idx for amp, phase and freq
                amp_r = self.amp_range[np.random.choice(self.env_[0])]
                phase_r = self.phase_range[np.random.choice(self.env_[1])]
                freq_r = self.freq_range[np.random.choice(self.env_[2])]
                switch_times.append(i)
            
            amp_range_list.append(amp_r)
            phase_range_list.append(phase_r)
            freq_range_list.append(freq_r) 
            # sample task from env
            amp = uniform_sample(amp_r)
            phase = uniform_sample(phase_r)
            freq = uniform_sample(freq_r)
            # sample batch data for each task
            for j in range(self.m_t):
                x_samp = uniform_sample(self.x_range)
                y_samp = amp*np.sin(freq*x_samp + phase) + self.noise_std*np.random.randn()
                x[i,j,:] = x_samp
                y[i,j,:] = y_samp

        if return_lists:
            return x,y,freq_range_list,amp_range_list,phase_range_list,switch_times

        return x,y
    
    def __getitem__(self,idx):
        # batching happens automatically
        
        data, labels, self.env_mu_list, self.amp_list, self.phase_list, self.switch_times = self.sample()
        switch_indicators = np.zeros(self.horizon)
        for i in range(self.horizon):
            if i in self.switch_times:
                switch_indicators[i] = 1.

        sample = {
            'x': data.astype(float),
            'y': labels.astype(float),
            'switch_times':switch_indicators.astype(float)
        }

        return sample

class SwitchingRotatedMNIST(Dataset):
    def __init__(self, args, transform=None, train=True, download=True):
 
        self.m_train = args.sample_num # train sample number for each task
        self.m_test = args.test_num # train sample number for each task
        self.root = os.path.expanduser(args.root)
        self.horizon = args.horizon# T
        self.transform = transform
        self.train = train# train only
        self.download = download
        self.switch_prob = args.hazard
        self.env_range = [[0,45], [45,90], [90,135], [135,180], [180, 225], [225, 270], [270, 325], [325,360]]
        self.switch_flag = []
        self.rot = []

        if self.train:
            self.train_data, self.train_labels  = self._get_data()
        else:
            self.train_data, self.train_labels, self.test_data, self.test_labels = self._get_data()


    def _get_data(self):
        """
            generate sequential tasks with env changing
        """
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root,
                                                                      train=True,
                                                                      download=self.download,
                                                                      transform=transforms.ToTensor()),
                                                       batch_size=60000,
                                                       shuffle=False)
        #shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
        for i, (x, y) in enumerate(train_loader):
            print(i,y)
            mnist_imgs = x
            mnist_labels = y
        
        if not self.train:
            test_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root,
                                                                      train=False,
                                                                      download=self.download,
                                                                      transform=transforms.ToTensor()),
                                                       batch_size=10000,
                                                       shuffle=False)

            for i, (x, y) in enumerate(test_loader):
                t_mnist_imgs = x
                t_mnist_labels = y
                test_images = []
                test_labels = []
        
        train_images = []
        train_labels = []
        
        for t in range(self.horizon):
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()

            if np.random.rand() < self.switch_prob or t == 0:
                env_idx = np.random.choice(len(self.env_range))# randomly choose a env range for rotation
                if t > 0:
                    if env_idx != last_env_idx:
                        self.switch_flag.append(1)
                last_env_idx = env_idx
            self.switch_flag.append(0)
            random_rot = uniform_sample(self.env_range[env_idx])
            self.rot.append(random_rot)
            # Run transforms
            mnist_trans_img = torch.zeros((self.m_train, 1, 28, 28))
            #shuffle idx of all the train data
            inds = np.arange(mnist_labels.size()[0])
            np.random.shuffle(inds)
          
            # choose m_train data from shuffled indx
            for i in range(self.m_train):
                #print(len(inds), random_rot, mnist_imgs[inds[i]])
                mnist_trans_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[inds[i]]), random_rot))
            train_images.append(mnist_trans_img)
            train_labels.append(mnist_labels[inds[:self.m_train]].cpu().detach().numpy()) 
            if not self.train:
                t_mnist_trans_img = torch.zeros((self.m_test, 1, 28, 28))
                #shuffle idx of all the train data
                inds = np.arange(t_mnist_labels.size()[0])
                np.random.shuffle(inds)
                # choose m_train data from shuffled indx
                for i in range(self.m_test):
                    # Run transforms
                    t_mnist_trans_img[i] = to_tensor(transforms.functional.rotate(to_pil(t_mnist_imgs[inds[i]]), random_rot))
                test_images.append(t_mnist_trans_img)
                test_labels.append(t_mnist_labels[inds[:self.m_test]].cpu().detach().numpy()) 
        # print(train_labels)
        # to onehot
        # y = torch.eye(10)
        # train_labels = y[train_labels]
        if not self.train:
            # test_labels = y[test_labels]
            return train_images, train_labels, test_images, test_labels
        
        return train_images, train_labels


    def __len__(self):
        # task numbers
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)

    def __getitem__(self, index):

        x_train = self.train_data[index]
        y_train = self.train_labels[index]
        if not self.train:
            x_test = self.test_data[index]
            y_test = self.test_labels[index]
            return x_train, y_train, x_test, y_test, self.switch_flag[index], self.rot[index]
        return x_train, y_train, self.switch_flag[index], self.rot[index]

def TEST_Gaussian(config):
    Traindataset = Switching2DGaussianDataset(config, True)
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    print(Traindataset[10])


def TEST_RotatedMNIST(config):
    # mnist = SwitchingRotatedMNIST(config)
    train_loader = DataLoader(
        SwitchingRotatedMNIST(config),
        batch_size=20,
        shuffle=False)
    # y_array = np.zeros((config.sample_num, 10))
    for i, (x, y, flag, rot) in enumerate(train_loader):
        # y_array += y.sum(dim=0).cpu().numpy()
        print(x.size())
        #if i == 0:
            # print(y)
        print(rot)
        n = max(x.size(0), 8)
        x = x.permute(1, 0, 2, 3, 4)
        print(x.size())
        comparison = torch.reshape(x, (-1, 1, 28, 28))#x.view(-1, 1, 28, 28)
        
        # save_image(comparison.cpu(),
        #                str(int(rot[0].cpu().detach().numpy())) + '_reconstruction_rotation_test.png',ncol=n)
    
        save_image(comparison.cpu(),
                        'reconstruction_rotation_test.png',nrow=n)

    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--horizon', type=int, help='horizon of tasks', default=20)
    argparser.add_argument('--hazard', type=int, help='hazard', default=0.5)
    argparser.add_argument('--test_num', type=int, help='test sample number per task', default=10)
    argparser.add_argument('--sample_num', type=int, help='sample number per task', default=8)
    argparser.add_argument('--sigma_eps', type=int, help='noise added on sinusoid', default=0.001)
    argparser.add_argument('--root', type=str, help='rotated mnist data path', default="./data/")
    config = argparser.parse_args()

    TEST_RotatedMNIST(config)
    