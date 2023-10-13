import numpy as np
import torch
import os
import torch.nn as nn
from pdb import set_trace
import random
from typing import List, Optional, Union
from collections import OrderedDict
from torchmeta.modules import MetaModule
import argparse


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def conv_block(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,3,padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                        )


def final_conv_block(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,3,padding=1),
#                         nn.BatchNorm2d(out_channels),
                        nn.MaxPool2d(2))


def listdir_nohidden(path):
    dir_list = []
    for f in os.listdir(path):
        if not f.startswith('.'):
             dir_list.append(f)
    return dir_list


def mask_nlls(y,likelihoods):
    """
    y: onehot labels: shape (..., n_classes)
    likelihood: per class: shape (..., n_classes)
    """
    # mask with y
    return torch.sum(y * likelihoods,dim=-1)


def compute_acc(y,nlls):
    # compute accuracy using nlls
    pred_class = torch.argmin(nlls,-1,keepdim=True)
    
    acc = y.gather(-1, pred_class).squeeze(-1)
    return acc


def get_prgx(config,horizon,batch_size,switch_times=None):

    model = config['model.model']
    sliding_window_len = config['data.window_length']

    if model == 'main' or model == 'conv_net':
        return None, None

    prgx = []
    task_sup = []
    last_switch = np.zeros(batch_size,dtype=int)

    for t in range(horizon):
        prgx_t = np.zeros((batch_size,t+1))
        task_supervision = np.zeros(batch_size)
        for i in range(batch_size):
            if model == 'sliding_window':
                prgx_t[i,max(t-sliding_window_len,0)] = 1
            elif model == 'no_task_change':
                prgx_t[i,t] = 1
            elif model == 'oracle':
                if switch_times[i,t] > 0.5:
                    last_switch[i] = t
                    if config['train.task_supervision'] is not None:
                        if np.random.rand() < config['train.task_supervision']:
                            task_supervision[i] = 1.
                            epsilon = 1e-5
                            prgx_t[i,:] = np.ones(t+1)*epsilon
                            prgx_t[i,last_switch[i]] = 1. - epsilon*t
                            
                            if config['train.oracle_hazard'] is not None:
                                raise NotImplementedError
                                
                if config['train.task_supervision'] is None:
                    if config['train.oracle_hazard'] is not None:
                        if last_switch[i] != 0:
                            prgx_t[i,0] = config['train.oracle_hazard']
                            prgx_t[i,last_switch[i]] = 1. - config['train.oracle_hazard']
                        else:
                            prgx_t[i,last_switch[i]] = 1.
                    else:
                        prgx_t[i,last_switch[i]] = 1.
                
            else:
                raise ValueError('make sure specified model is implemented')


        prgx_t = torch.tensor(prgx_t).float()
        task_supervision = torch.tensor(task_supervision).float()


        if config['data.cuda'] >= 0:
            prgx_t = prgx_t.cuda(config['data.cuda'])
            task_supervision = task_supervision.cuda(config['data.cuda'])

        
        prgx.append(prgx_t)
        task_sup.append(task_supervision)

    if config['train.task_supervision'] is None:
        return prgx, None
    else:
        return prgx, task_sup
    

def str2bool(value: Union[str, bool]) -> bool:
    """Parses a `bool` value from a string.

    Can be used as the `type` argument to the `add_argument()` function for easy
    parsing of flags/boolean values.

    Taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Examples:
    >>> str2bool("1")
    True
    >>> str2bool("0")
    False
    >>> str2bool("false")
    False
    >>> str2bool("true")
    True
    >>> str2bool("yes")
    True
    >>> str2bool("no")
    False
    """
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected for argument, received '{value}'")

###### maml utils #######
# This function does not change params in model, only output updated params
def update_parameters(model, loss, params=None, step_size=0.5, first_order=False,
            freeze_visual_features=False, no_meta_learning=False):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.

    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.

    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.

    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.
    """
    if not isinstance(model, MetaModule):
        raise ValueError()

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss, params.values(),
        create_graph=not first_order, allow_unused=True)

    out = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            if (freeze_visual_features and 'visual_features' in name) or no_meta_learning:
                out[name] = param
            else:
                out[name] = param - step_size[name] * grad
    else:
        for (name, param), grad in zip(params.items(), grads):
            if (freeze_visual_features and 'visual_features' in name) or no_meta_learning:
                out[name] = param
            else:
                out[name] = param - step_size * grad

    return out



def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()


def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()


def set_seed(args, manualSeed):
    #####seed#####
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    if args.device != "cpu":
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    ######################################################


def is_connected(host='http://google.com'):
    import urllib.request
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False


class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.

    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'


def wandb_wrapper(args, wandb=None, run=0):
    if wandb is None:
        if not is_connected():
            print('no internet connection. Going in dry')
            os.environ['WANDB_MODE'] = 'dryrun'
        import wandb
        if args.wandb_key is not None:
            wandb.login(key=args.wandb_key)

    if args.name is None:
        if "DCML" == args.model_name_impv:
            args.name = args.dataset + "_" + args.model_name_impv \
                + "_eta0_" + str(args.eta_0) + "_epsilon0_" + str(args.epsilon_0_rate) \
                + "_rho_" + str(args.rho) + "_L_" + str(args.L) + "_K_" + str(args.K) \
                + "_switch_method_" + str(args.switch_method) + "_win_" + str(args.win) \
                + "_D_hat_" + str(args.D_hat_guess)
        elif "CMAML_hyper" == args.model_name_impv:
            args.name = args.dataset + "_" + args.model_name_impv \
                + "_meta_lr_" + str(args.meta_lr) + "_num_steps_" + str(args.num_steps) \
                + "_step_size_" + str(args.step_size) + "_env_thres_" \
                + str(args.cl_strategy_thres) + "_task_thres_" + str(args.cl_tbd_thres)
        else:
            args.name = args.dataset + "_" + args.model_name_impv

    wandb.init(project=args.wandb, name=args.name, group=args.group, reinit=True)
    wandb.config.update(args)
    return wandb
