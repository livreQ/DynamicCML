#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Tue Dec 27 13:37:49 2022
# File Name: osaka_datasets.py
# Description: Code adapted from https://github.com/ServiceNow/osaka.git
"""

import sys
import torch
import numpy as np
from pdb import set_trace
from src.utils.omniglot import Omniglot
from torchvision.datasets import MNIST, FashionMNIST
from src.utils.tiered_imagenet import NonEpisodicTieredImagenet
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Optional, Union

# --------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------

def select_from_tensor(tensor, index):
    """ equivalent to tensor[index] but for batched / 2D+ tensors """

    last_dim = index.dim() - 1

    assert tensor.dim() >= index.dim()
    assert index.size()[:last_dim] == tensor.size()[:last_dim]

    # we have to make `train_idx` the same shape as train_data, or else
    # `torch.gather` complains.
    # see https://discuss.pytorch.org/t/batched-index-select/9115/5

    missing_dims = tensor.dim() - index.dim()
    index = index.view(index.size() + missing_dims * (1,))
    index = index.expand((-1,) * (index.dim() - missing_dims) + tensor.size()[(last_dim+1):])

    return torch.gather(tensor, last_dim, index)

def order_and_split(data_x, data_y):
    """ given a dataset, returns (num_classes, samples_per_class, *data_x[0].size())
        tensor where samples (and labels) are ordered and split per class """

    # sort according to the label
    out_train = [
        (x,y) for (x,y) in sorted(zip(data_x, data_y), key=lambda v : v[1]) ]

    # stack in increasing label order
    data_x, data_y = [
            torch.stack([elem[i] for elem in out_train]) for i in [0,1] ]

    # find first indices of every class
    n_classes = data_y.unique().size(0)
    idx       = [((data_y + i) % n_classes).argmax() for i in reversed(range(n_classes))]
    #print(n_classes, idx)
    idx       = idx + [len(data_y)]#[0] + [x + 1 for x in sorted(idx)]

    # split into different classes
    to_chunk = [a - b for (a,b) in zip(idx[1:], idx[:-1])]
    data_x   = data_x.split(to_chunk)
    data_y   = data_y.split(to_chunk)

    # give equal amt of points for every class
    #TODO(if this is restrictive for some dataset, we can change)
    min_amt  = min([x.size(0) for x in data_x])
    data_x   = torch.stack([x[:min_amt] for x in data_x])
    data_y   = torch.stack([y[:min_amt] for y in data_y])

    # sanity check
    for i, item in enumerate(data_y):
        assert item.unique().size(0) == 1 and item[0] == i, 'wrong result'

    return data_x, data_y

# --------------------------------------------------------------------------
# Datasets and Streams (the good stuff)
# --------------------------------------------------------------------------

class MetaDataset(Dataset):
    """ Dataset similar to BatchMetaDataset in TorchMeta """

    def __init__(self, train_data, val_data,  m_tr, m_va, n_way,
                    args=None, **kwargs):

        '''
        Parameters
        ----------

        train_data : Array of (x,) pairs, one for each class. Contains all the
            training data that should be available at meta-training time (inner loop).
        test_data  : Array of (x,) pairs, one for each class. These are the
            same classes as in `train_data`. Used at meta-testing time (outer loop).
        n_way      : number of classes per task at meta-testing
        m_tr : number of samples per classes
        m_va : number of samples per classes

        '''

        # NOTE: for now assume train_data and test_data have shape
        # (n_classes, n_samples_per_task, *data_shape).

        # separate the classes into tasks  
        self._len        = None
        self.n_way       = n_way # number of classes for each task
        self.kwargs      = kwargs
        self.n_classes   = len(train_data)# number of all classes

        self.m_tr  = m_tr # number of samples for meta-train-train task
        self.m_va  = m_va # number of samples for meta-train-val task

        if args is None:
            self.input_size  = [28, 28]
            self.device      = 'cpu'
            self.is_classification_task = True
        else:
            self.input_size  = args.input_size
            self.device      = args.device
            self.is_classification_task = args.is_classification_task

        self.all_classes = np.arange(self.n_classes)

        self.train_data  = train_data
        self.val_data   = val_data

        if args.dataset == 'tiered-imagenet':
            self.cpu_dset = True
        else:
            self.cpu_dset = False

    def __len__(self):
        # return the number of train / test batches that can be built
        # without sample repetition
        if self._len is None:
            n_samples = sum([x.shape[0] for x in self.train_data])
            self._len = n_samples // (self.n_way * (self.m_tr + self.m_va))

        return self._len


    def __getitem__(self, index):
        if self.is_classification_task:
            return self._getitem_classification(index)
        else:
            return self._getitem_regression(index)

    def _getitem_regression(self, index):
        train_x = self.train_data[..., 0, None]
        train_y = self.train_data[..., 1, None]
        val_x = self.val_data[..., 0, None]
        val_y = self.val_data[..., 1, None]

        if self.cpu_dset:
            train_x = train_x.to(self.device)
            train_y = train_y.to(self.device)
            val_x = val_x.to(self.device)
            val_y = val_y.to(self.device)

        return { "train": [train_x, train_y], "val": [val_x, val_y],}

    def _getitem_classification(self, index):
        # NOTE: This method COMPLETELY ignores the index. This will be a problem
        # if you wish to recover a specific batch of data.

        classes_in_task = np.random.choice(self.all_classes, self.n_way, replace=False)
        train_samples_in_class = self.train_data.shape[1]
        val_samples_in_class  = self.val_data.shape[1]

        train_data = self.train_data[classes_in_task]
        val_data  = self.val_data[classes_in_task]

        # sample indices for meta train
        train_idx = torch.Tensor(self.n_way, self.m_tr)
        if not(self.cpu_dset):
            train_idx = train_idx.to(self.device)
        train_idx = train_idx.uniform_(0, train_samples_in_class).long()

        # samples indices for meta validation
        val_idx = torch.Tensor(self.n_way, self.m_va)
        if not(self.cpu_dset):
            val_idx = val_idx.to(self.device)
        val_idx = val_idx.uniform_(0, val_samples_in_class).long()

        train_x = select_from_tensor(train_data, train_idx)
        val_x  = select_from_tensor(val_data,  val_idx)

        train_x = train_x.view(-1, *self.input_size)
        val_x = val_x.view(-1, *self.input_size)

        # build label tensors
        train_y = torch.arange(self.n_way).view(-1, 1).expand(-1, self.m_tr)
        train_y = train_y.flatten()

        val_y  = torch.arange(self.n_way).view(-1, 1).expand(-1, self.m_va)
        val_y  = val_y.flatten()

        if self.cpu_dset:
            train_x = train_x.float().to(self.device)
            train_y = train_y.to(self.device)
            val_x = val_x.float().to(self.device)
            val_y = val_y.to(self.device)

        #return train_x, train_y, test_x, test_y

        # same signature are TorchMeta
        out = {}
        out['train'], out['val'] = [train_x, train_y], [val_x, val_y]
        return out



class ContinualMetaDataset(Dataset):
    """ stream of tasks in dynamic task environment """

    def __init__(self, cl_envs, n_shots=1, n_shots_test =1,
            n_way=5, prob_statio=.8, prob_env_switch=.1, prob_envs=[0.1, 0.8, 0.1], env_names=['env0', 'env1', 'env2'], args=None, **kwargs):

        '''
        Parameters
        ----------

        train_data : Array of (x,) pairs, one for each class. Contains the SAME
            classes used during (meta) training, but different samples.
        test_data  : Array of (x,) pairs, one for each class. These are DIFFERENT
            classes from the ones used during (meta) training.
        n_way      : number of classes per task at cl-test time
        n_shots    : number of samples per classes at cl-test time
        prob_envs : [prob_train=0.1, prob_test=0.8, prob_ood=0.1]

        '''

        
        if args.dataset == 'tiered-imagenet':
            self.cpu_dset = True
        else:
            self.cpu_dset = False
        self.n_shots    = n_shots
        self.n_shots_test = n_shots_test
        self.n_way      = n_way
        self.envs    = env_names
        self.envs_id = list(range(len(self.envs)))
        self.probs    = np.array(prob_envs)
        assert np.sum(self.probs) == 1.
        self.data     = cl_envs #[pretrain_data, ood_data1, ood_data2]
        self.p_statio = prob_statio
        self.p_env_switch = prob_env_switch
        self.task_sequence: List[str] = []
        self.n_steps_per_task = 1
        self.index_in_task_sequence = 0
        self.steps_done_on_task = 0

        if args is None:
            self.input_size  = [28,28]
            self.device      = 'cpu'
            self.is_classification_task = True
        else:
            self.input_size  = args.input_size
            self.device      = args.device
            self.is_classification_task = args.is_classification_task
            self.task_sequence = args.task_sequence
            self.n_steps_per_task = args.n_steps_per_task

        self.env_name_map = dict(zip(self.envs, self.envs_id))

        # env in which to start ( 0 --> 'train' )
        self._env = 0
        self._classes_in_task = None
        self._samples_in_class = None


    def __len__(self):
        # this is a never ending stream
        return sys.maxsize


    def __getitem__(self, index):
        if self.is_classification_task:
            return self._getitem_classification(index)
        else:
            return self._getitem_regression(index)

    def _getitem_regression(self, index):
        task_switch = False
        if self.task_sequence:
            self.steps_done_on_task += 1

            if self.steps_done_on_task >= self.n_steps_per_task:
                task_switch = True
                self.steps_done_on_task = 0
                self.index_in_task_sequence += 1
                self.index_in_task_sequence %= len(self.task_sequence)

            env_name = self.task_sequence[self.index_in_task_sequence]
            self._env = self.env_name_map[env_name]
        else:
            if (np.random.uniform() > self.p_statio):
                env  = np.random.choice(self.envs_id, p=self.probs)
                self._env = env
                task_switch = env != self._env

        env_data = self.data[self._env]

        x = env_data[..., 0, None]
        y = env_data[..., 1, None]
        if self.cpu_dset:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y, task_switch, self.envs[self._env]

    def _getitem_classification(self, index):
        # NOTE: This method COMPLETELY ignores the index. This will be a problem
        # if you wish to recover a specific batch of data.

        # NOTE: using multiple workers (`num_workers > 0`) or `batch_size  > 1`
        # will have undefined behaviour. This is because unlike regular datasets
        # here the sampling process is sequential.
        task_switch = 0
        env_switch = 0
        if self.task_sequence:
            self.steps_done_on_task += 1

            if self.steps_done_on_task >= self.n_steps_per_task:
                task_switch = 1
                self.steps_done_on_task = 0
                self.index_in_task_sequence += 1
                self.index_in_task_sequence %= len(self.task_sequence)

            env_name = self.task_sequence[self.index_in_task_sequence]
            self._env = self.env_name_map[env_name]
        elif (np.random.uniform() > self.p_statio) or (self._classes_in_task is None):
            # env  = np.random.choice(self.envs_id, p=self.probs)
            # self._env = env
            # task_switch = env != self._env
            # TODO: this makes a switch even if staying in same env!
            task_switch = 1
            if (np.random.uniform() < self.p_env_switch):
                new_env = np.random.choice(self.envs_id, p=self.probs)
                if self._env != new_env:
                    self._env = new_env
                    env_switch = 1

            env_data = self.data[self._env]
            n_classes = len(env_data)
            self._samples_in_class = env_data.size(1)

            # sample `n_way` classes
            self._classes_in_task = np.random.choice(np.arange(n_classes), self.n_way,
                    replace=False)

        else:

            task_switch = 0

        env_data = self.data[self._env]
        data = env_data[self._classes_in_task]

        # sample indices for meta train
        idx = torch.Tensor(self.n_way, self.n_shots)#.to(self.device)
        idx = idx.uniform_(0, self._samples_in_class).long() # how many samples in each tasks
        if not(self.cpu_dset):
            idx = idx.to(self.device)
        data = select_from_tensor(data, idx)
        # build label tensors
        labels = torch.arange(self.n_way).view(-1, 1).expand(-1, self.n_shots).to(self.device)
        # squeeze
        data = data.view(-1, *self.input_size)
        labels = labels.flatten()
        if self.cpu_dset:
            data = data.float().to(self.device)
            labels = labels.to(self.device)
        out = {}
        out['train'] = [data, labels]
        #======= test data =======
        if self.n_shots_test != 0:
            idx = torch.Tensor(self.n_way, self.n_shots_test)#.to(self.device)
            idx = idx.uniform_(0, self._samples_in_class).long() # how many samples in each tasks
            if not(self.cpu_dset):
                idx = idx.to(self.device)
            data = env_data[self._classes_in_task]
            data = select_from_tensor(data, idx)
            # build label tensors
            labels = torch.arange(self.n_way).view(-1, 1).expand(-1, self.n_shots_test).to(self.device)
            # squeeze
            data = data.view(-1, *self.input_size)
            labels = labels.flatten()
            if self.cpu_dset:
                data = data.float().to(self.device)
                labels = labels.to(self.device)
            out['val'] = [data, labels]
        else:
            # same as train but won't be used
            out['val'] = [data, labels]
        return out, task_switch, env_switch, self.envs[self._env]


def init_dataloaders(args):

    if args.dataset == 'omniglot':
  
        args.is_classification_task = True
        args.prob_envs = [0.5, 0.25, 0.25]
        args.n_train_cls = 900
        args.n_test_cls = 100
        # can be resampled with MetaDataset, split to train, val
        args.n_train_samples = 10  

        args.input_size = [1,28,28]
        Omniglot_dataset = Omniglot(args.folder).data
        Omniglot_dataset = torch.from_numpy(Omniglot_dataset).type(torch.float).to(args.device)
        meta_train_dataset = Omniglot_dataset[:args.n_train_cls]
        meta_train_train = meta_train_dataset[:,:args.n_train_samples,:,:] # suppport
        meta_train_val = meta_train_dataset[:,args.n_train_samples:,:,:]   # query
        
        meta_test_dataset = Omniglot_dataset[args.n_train_cls : (args.n_train_cls + args.n_test_cls)]
        meta_test_train = meta_test_dataset[:,:args.n_train_samples,:,:] 
        meta_test_val = meta_test_dataset[:,args.n_train_samples:,:,:]

        omn = Omniglot_dataset
        mnist = MNIST(args.folder, train=True,  download=True)
        fmnist = FashionMNIST(args.folder, train=True,  download=True)
        mnist, _ = order_and_split(mnist.data, mnist.targets)
        fmnist, _ = order_and_split(fmnist.data, fmnist.targets)
        mnist = mnist[:,:,None,:,:].type(torch.float).to(args.device)
        fmnist = fmnist[:,:,None,:,:].type(torch.float).to(args.device)
        cl_envs = [omn, mnist, fmnist]
        cl_env_names = ["omniglot", "mnist", "fasion_mnist"]

    elif args.dataset == "tiered-imagenet":
    
        args.prob_envs = [0.3, 0.3, 0.4]
        args.is_classification_task = True
        args.n_train_cls = 100
        args.n_val_cls = 100
        args.n_train_samples = 500

        args.input_size = [3,64,64]
        args.folder = args.folder + "/" + args.dataset
        tiered_dataset = NonEpisodicTieredImagenet(args.folder, split="train")

        meta_train_dataset = tiered_dataset.data[:args.n_train_cls]
        meta_train_train = meta_train_dataset[:,:args.n_train_samples, ...]
        meta_train_val = meta_train_dataset[:,args.n_train_samples:,...]

        meta_test_dataset = tiered_dataset.data[args.n_train_cls : (args.n_train_cls+args.n_val_cls)]
        meta_test_train = meta_test_dataset[:,:args.n_train_samples,:,:]
        meta_test_val = meta_test_dataset[:,args.n_train_samples:,:,:]

        cl_env0 = tiered_dataset.data
        cl_env1 = tiered_dataset.data[(args.n_train_cls + args.n_val_cls):].type(torch.float)
        # real ood
        cl_env2 = NonEpisodicTieredImagenet(args.folder, split="val").data.type(torch.float)
        cl_envs = [cl_env0, cl_env1, cl_env2]
        cl_env_names = ["tired0", "tired1", "tired2"]

    elif args.dataset == "synbols":    
        args.is_classification_task = True
        args.prob_envs = [0.5, 0.25, 0.25]
        args.input_size = [3,32,32]
        path = os.path.join(args.folder, 'cl-synbols_trn-sz1000_32x32')
        train = torch.from_numpy(np.load(os.path.join(path, 'trn.npy'))).permute(0,1,4,2,3).float().to(args.device)
        new_alphabet = torch.from_numpy(np.load(os.path.join(path, 'new_alphabet.npy'))).permute(0,1,4,2,3).float().to(args.device)
        font = torch.from_numpy(np.load(os.path.join(path, 'font_task.npy'))).permute(0,1,4,2,3).float().to(args.device)
        meta_train_dataset = train
        print('shape of train:', train.shape)
        # meta_train_train = meta_train_dataset[:, :args.num_shots, ...] # args.num_shots used for num samples
        # meta_train_test = meta_train_dataset[:, args.num_shots:, ... ]
        meta_train_train = meta_train_dataset[:, :600, ...]
        meta_train_val = meta_train_dataset[:, 600:, ...]

        meta_test_dataset = new_alphabet
        print('shape of val:', new_alphabet.shape)
        # meta_val_train = meta_val_dataset[:,:(args.num_shots+1), ...]
        # meta_val_test = meta_val_dataset[:,(args.num_shots+1):, ...]
        meta_test_train = meta_test_dataset[:, :600, ...]
        meta_test_val = meta_test_dataset[:, 600:, ...]

        # if args.env=='train':
        cl_envs = [train, new_alphabet, font]
        cl_env_names = ["origin", "new_alphabet", "font"]
        
    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(args.dataset))
    
    #*************  Create Meta Pretrain DataLoader ******************
    meta_train_dataloader = MetaDataset(meta_train_train, meta_train_val, args=args,
            m_tr=args.m_tr, m_va=args.m_va, n_way=args.n_ways)
    meta_train_dataloader = DataLoader(meta_train_dataloader, batch_size=args.pretrain_batch_size)

    
    meta_test_dataloader = MetaDataset(meta_test_train, meta_test_val, args=args,
            m_tr=args.m_tr, m_va=args.m_va, n_way=args.n_ways)
    meta_test_dataloader = DataLoader(meta_test_dataloader, batch_size=args.pretrain_batch_size)

    #************  Create Continual Meta DataLoader ******************
    cl_dataloader = ContinualMetaDataset(cl_envs, n_shots=args.n_shots, n_shots_test=args.n_shots_test, n_way=args.n_ways, 
            prob_statio=args.prob_statio, prob_env_switch=args.prob_env_switch, 
            env_names=cl_env_names, prob_envs=args.prob_envs, args=args)
    cl_dataloader = DataLoader(cl_dataloader, batch_size=1)#one few-shot task at a time

    del meta_train_dataset, meta_train_train, meta_train_val, meta_test_dataset,\
            meta_test_train, meta_test_val, cl_envs

    return meta_train_dataloader, meta_test_dataloader, cl_dataloader




