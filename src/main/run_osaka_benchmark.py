#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Tue Dec 27 13:07:32 2022
# File Name: run_osaka_benchmark.py
# Description: Code adapted from https://github.com/ServiceNow/osaka.git
"""

import torch
import torch.nn.functional as F
import math
import os
import time
import json
import random
import logging
import numpy as np
import copy
from pdb import set_trace

from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Compose
from src.dataloader.osaka_datasets import init_dataloaders
from src.learners.osaka_model import ModelConvSynbols, ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from src.learners.maml import MAML, ModularMAML
from src.learners.dmogd import DMOGD
from src.utils.utils import ToTensor1D, set_seed, is_connected, wandb_wrapper
#from torchsummary import summary
import argparse
from typing import List, Optional, Union
import yaml
from src.utils.utils import str2bool
#from src.utils.utils import wandb_wrapper
from args import parse_args
from src.learners.model import Model
import uuid
import gc


def boilerplate(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gc.collect()
    torch.cuda.empty_cache()
    set_seed(args, args.seed)
    if args.model_name_impv is None:
        args.group = str(uuid.uuid1())
    else:
        args.group = args.model_name_impv + '_' + str(args.prob_statio)
    #args.name = args.model_name_impv + '_'
    #args.group = args.model_name + "_" + str(args.prob_statio) + "_" + str(args.prob_statio)
    return args

def init_models(args, metalearner=None):
    """
        init models for osaka datasets
    """
    pretrain_model_loaded = False
    if not metalearner is None:
        model = metalearner.model
    else:
        if args.dataset == 'omniglot':
            model = ModelConvOmniglot(args.n_ways, hidden_size=args.hidden_size, deeper=args.deeper)
            loss_function = F.cross_entropy
        if args.dataset == 'tiered-imagenet':
            model = ModelConvMiniImagenet(args.n_ways, hidden_size=args.hidden_size, deeper=args.deeper)
            loss_function = F.cross_entropy
        if args.dataset == 'synbols':
            model = ModelConvSynbols(args.n_ways, hidden_size=args.hidden_size, deeper=args.deeper)
            loss_function = F.cross_entropy
        if args.dataset == "harmonics":
            model = ModelMLPSinusoid(hidden_sizes=[40, 40])
            loss_function = F.mse_loss
        #---------
        # print(model)
        # print(summary(model, (1, 28, 28)))
        # print(model.config)
        # if args.model_name == 'DCML':
        #     # our methods
        #     model = Model(model.config)
        if args.pretrain and os.path.exists(args.output + "/" + args.pretrain_model):
            #contain a pretrain phase and already exists a pretrained model
            model.load_state_dict(torch.load(args.output + "/" + args.pretrain_model))
            pretrain_model_loaded = True
        print(model)
        # print(summary(model, (1, 28, 28)))
    if metalearner is None:
        if args.method == 'MAML':
            metalearner = MAML(model, loss_function, args)
        elif args.method == 'ModularMAML':
            if args.wandb_key is not None:
               wandb = wandb_wrapper(args)
            metalearner = ModularMAML(model, loss_function, args, wandb=wandb)
        elif args.method == 'DMOGD':
            metalearner = DMOGD(args, model.config, model).to(args.device)
    #print("========== model parameters ========")
    #print(metalearner.model.parameters())
    #print(metalearner.model.state_dict())
    return (metalearner, pretrain_model_loaded)

def pretraining(args, metalearner, meta_train_dataloader, meta_test_dataloader, wandb=None):
    """
        pretrain process
    """
    if not os.path.exists(args.output + "/" + args.pretrain_model):
        # best_metalearner = copy.deepcopy(metalearner)
        print("model not exist")
        if args.wandb_key is not None:
            wandb = wandb_wrapper(args, wandb)
        best_metalearner = metalearner
        if args.n_pretrain_epochs==0:
            pass
        else:
            best_test = 0.
            epochs_overfitting = 0
            epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.n_pretrain_epochs)))
            for epoch in range(args.n_pretrain_epochs):
                # meta train, see n_pretrain_batches * pretrain_batch_size tasks
                metalearner.meta_train(meta_train_dataloader, max_batches=args.n_pretrain_batches,
                                    verbose=args.verbose, desc='Training', leave=False)
                # meta test
                results = metalearner.meta_test(meta_test_dataloader, max_batches=args.n_pretrain_batches/10,
                                                verbose=args.verbose, epoch=epoch,
                                                desc=epoch_desc.format(epoch + 1))
                result_test = results['evas_after']
                print("train performance:{:.4f}, test performance:{:.4f}".format(results['evas_before'], results['evas_after']))
                if wandb is not None:
                    wandb.log({'Slow weights test performance:': results['evas_before']}, step=epoch)
                    wandb.log({'Fast weights test performance:': results['evas_after']}, step=epoch)
                # early stopping:
                if (best_test is None) or (best_test < result_test):
                    epochs_overfitting = 0
                    best_val = result_test
                    best_metalearner = copy.deepcopy(metalearner)
                    if args.output is not None:
                        with open(args.output + "/" + args.pretrain_model, 'wb') as f:
                            torch.save(best_metalearner.model.state_dict(), f)
                else:
                    epochs_overfitting +=1
                    if epochs_overfitting > args.patience:
                        break

            print('\npretraining done!\n')
            if wandb is not None:
                wandb.log({'best_val':best_val}, step=epoch)
    else:
        best_metalearner = copy.deepcopy(metalearner)

    cl_model = best_metalearner#copy.deepcopy(best_metalearner)

    del metalearner#, best_metalearner

    return cl_model


def wandb_log(env, run_envs, online_scores, test_scores, task_switch, env_switch, true_task_switch, true_env_switch,\
              tbd_precisions, tbd_recalls, ecd_precisions, ecd_recalls, eva_metrics, t, run,\
                 last_time_step=False, final_run=False, wandb=None):
    # task boundary detection performance at current timestep
    wandb.log({"current_task_boundary":true_task_switch}, step=t)
    wandb.log({"current_env_switch":true_env_switch}, step=t)
    wandb.log({"pred_task_boundary":task_switch}, step=t)
    wandb.log({"pred_env_switch":env_switch}, step=t)
    wandb.log({"tbd_precision_so_far":np.mean(tbd_precisions)}, step=t)
    wandb.log({"tbd_recall_so_far":np.mean(tbd_recalls)}, step=t)
    wandb.log({"ecd_precision_so_far":np.mean(ecd_precisions)}, step=t)
    wandb.log({"ecd_recall_so_far":np.mean(ecd_recalls)}, step=t)

    wandb.log({"current_acc":online_scores[-1]}, step=t)
    wandb.log({"current_acc_{}".format(env):online_scores[-1]}, step=t)
    wandb.log({"current_test_acc":online_scores[-1]}, step=t)
    wandb.log({"current_test_acc_{}".format(env):test_scores[-1]}, step=t)

    wandb.log({"avg_acc_so_far":np.mean(online_scores)}, step=t)
    wandb.log({"avg_test_acc_so_far":np.mean(test_scores)}, step=t)
    for env in set(run_envs):
        wandb.log({"avg_acc_so_far_{}".format(env):
                    np.mean(np.array(online_scores)[np.array(run_envs)==env])}, step=t)
        wandb.log({"avg_test_acc_so_far_{}".format(env):
                    np.mean(np.array(test_scores)[np.array(run_envs)==env])}, step=t)
    ## for smoothness sake, logging avg accuracy over 100 steps
    if t % 100 == 0 and t > 0:
        wandb.log({"online_acc_smooth":np.mean(online_scores[-100:])}, step=t)
        wandb.log({"online_test_acc_smooth":np.mean(test_scores[-100:])}, step=t)
        for env in set(run_envs):
            wandb.log({"online_acc_smooth_{}".format(env):
                    np.mean(np.array(online_scores[-100:])[np.array(run_envs[-100:])==env])}, step=t)
            wandb.log({"online_test_acc_smooth_{}".format(env):
                    np.mean(np.array(test_scores[-100:])[np.array(run_envs[-100:])==env])}, step=t)

    if last_time_step:
        run_eva = np.mean(online_scores)
        wandb.log({'per_run_final_acc':run_eva}, step=run)
        eva_metrics['final_accs'].append(run_eva)
        run_test_eva = np.mean(test_scores)
        wandb.log({'per_run_final_test_acc':run_test_eva}, step=run)
        eva_metrics['final_test_accs'].append(run_test_eva)
        for env in set(run_envs):
            run_eva = np.mean(np.array(online_scores)[np.array(run_envs)==env])
            wandb.log({'per_run_final_acc_{}'.format(env):run_eva}, step=run)
            run_test_eva = np.mean(np.array(test_scores)[np.array(run_envs)==env])
            wandb.log({'per_run_final_test_acc_{}'.format(env):run_test_eva}, step=run)
            if env not in eva_metrics:
                eva_metrics[env] = []
                eva_metrics[env + "_test"] = []
            eva_metrics[env].append(run_eva)
            eva_metrics[env + "_test"].append(run_test_eva)
        #======= log task boudary detection evaluations =======
        per_run_tbd_precision = np.mean(tbd_precisions)
        per_run_tbd_recall = np.mean(tbd_recalls)
        wandb.log({'per_run_tbd_precision':per_run_tbd_precision}, step=t)
        wandb.log({'per_run_tbd_recall':per_run_tbd_recall}, step=t)
        eva_metrics['tbd_precisions'].append(per_run_tbd_precision)
        eva_metrics['tbd_recalls'].append(per_run_tbd_recall)
        per_run_tbd_f1_score = 2 * (per_run_tbd_precision*per_run_tbd_recall)/(per_run_tbd_precision+per_run_tbd_recall)
        wandb.log({'per_run_tbd_f1_score':per_run_tbd_f1_score}, step=t)
        eva_metrics['tbd_f1_scores'].append(per_run_tbd_f1_score)

        #======== log env change detection evaluations ========
        per_run_ecd_precision = np.mean(ecd_precisions)
        per_run_ecd_recall = np.mean(ecd_recalls)
        wandb.log({'per_run_ecd_precision':per_run_ecd_precision}, step=t)
        wandb.log({'per_run_ecd_recall':per_run_ecd_recall}, step=t)
        eva_metrics['ecd_precisions'].append(per_run_ecd_precision)
        eva_metrics['ecd_recalls'].append(per_run_ecd_recall)
        per_run_ecd_f1_score = 2 * (per_run_ecd_precision*per_run_ecd_recall)/(per_run_ecd_precision + per_run_ecd_recall)
        wandb.log({'per_run_ecd_f1_score':per_run_ecd_f1_score}, step=t)
        eva_metrics['ecd_f1_scores'].append(per_run_ecd_f1_score)

    if not final_run:
        if last_time_step:
            wandb.join()
    else:
        for key in eva_metrics:
            avg = np.mean(eva_metrics[key])
            std = np.std(eva_metrics[key])
            print(f'{key}: \t {avg:.2f} +/- {std:.2f}')
            wandb.log({"{}_avg".format(key):avg})
            wandb.log({"{}_std".format(key):std})
        #wandb.log({'best_pretrain_val':cl_model.best_pretrain_val})


def continual_learning(args, cl_model, cl_dataloader, wandb=None):
    """
        continual learning process
    """
    print("======= wandb group ======: ", args.group)
    # record results for all envs
    iftest =  args.n_shots_test > 0
    eva_metrics = {}#dict(zip(envs, [[], [], []]))
    # avg evaluation at final timestep for each run
    eva_metrics['final_accs'] = []
    eva_metrics['final_test_accs'] = []
    eva_metrics['tbd_precisions'], eva_metrics['tbd_recalls'], eva_metrics['tbd_f1_scores'] = [], [], []
    eva_metrics['ecd_precisions'], eva_metrics['ecd_recalls'], eva_metrics['ecd_f1_scores'] = [], [], []
    #is_classification_task = args.task == "classification"
    for run in range(args.n_runs):
        #set_seed(args, args.seed) if run==0 else set_seed(args, random.randint(0,100000))
        print("========= run count ======== : ", run)
        if args.wandb_key is not None:
            wandb = wandb_wrapper(args, wandb, run)
        online_scores, test_scores = [], []
        tbd_precisions, tbd_recalls = [10e-8], [10e-8]
        ecd_precisions, ecd_recalls = [10e-8], [10e-8]
        run_envs= []
        online_scores_env, test_scores_env = {}, {} #= dict(zip(envs, [[], [], []])) #dict(zip(envs, [[], [], []]))
        ## for each run init model from pretrained model
        cl_model, pretrain_model_loaded = init_models(args, cl_model)
        for t, batch in enumerate(cl_dataloader):
            # one task per time
            data, true_task_switch, true_env_switch, env = batch
            #print(data['train'][0].shape, data['val'][0].shape)
            if args.cl_accumulate:
                curr_results = cl_model.observe_accumulate(batch)
            else:
                curr_results = cl_model.observe(batch)
            ## Reporting:
            current_env = env[0]
            online_score = curr_results["eva"]
            if iftest:
                test_score = curr_results["eva_test"]
            else:
                test_score = 0
            run_envs.append(current_env)
            online_scores.append(online_score)
            test_scores.append(test_score)
            if current_env not in online_scores_env:
                online_scores_env[current_env] = []#.append(online_score)
                test_scores_env[current_env] = [] #append(test_score)

            online_scores_env[current_env].append(online_score)
            test_scores_env[current_env].append(test_score)

            env_switch = float(curr_results['env_switch'])
            task_switch = float(curr_results['task_boundary'])
            tbd_score = float(task_switch == true_task_switch)
            ecd_score = float(env_switch == true_env_switch)
            if task_switch:
                # True Positive and  False Positive
                tbd_precisions.append(tbd_score)
            if true_task_switch:
                # True Positive and False Negative
                tbd_recalls.append(tbd_score)
            if env_switch:
                ecd_precisions.append(ecd_score)
            if true_env_switch:
                ecd_recalls.append(ecd_score)

            if args.verbose and (t % 100 == 0):
                avg_eva_so_far = np.mean(online_scores)
                avg_test_eva_so_far = np.mean(test_scores)

                tbd_precision_so_far = np.mean(tbd_precisions)
                tbd_recall_so_far = np.mean(tbd_recalls)

                ecd_precision_so_far = np.mean(ecd_precisions)
                ecd_recall_so_far = np.mean(ecd_recalls)

                message = []
                print(
                    f"run: {run}",
                    f"current env: {current_env}",
                    f"Avg Online Evaluation So Far: {avg_eva_so_far:.4f}",
                    f"Avg Test Evaluation So Far: {avg_test_eva_so_far:.4f}",
                    f"Current Task Boundary Detection Precision: {tbd_precision_so_far:.2f}",
                    f"Current Task Boundary Detection Recall: {tbd_recall_so_far:.2f}",
                    f"Current Env Change Detection Precision: {ecd_precision_so_far:.2f}",
                    f"Current Env Change Detection Recall: {ecd_recall_so_far:.2f}",
                    f"timestep: {t}", sep="\t"
                )
                if args.model_name == "DCML":
                    cl_model.show_learner_params()

            last_time_step = t == args.timesteps-1
            final_run = run == args.n_runs -1
            if wandb is not None:
                wandb_log(current_env, run_envs, online_scores, test_scores, task_switch, env_switch, true_task_switch, true_env_switch,\
                         tbd_precisions, tbd_recalls, ecd_precisions, ecd_recalls, eva_metrics, t, run, last_time_step, final_run, wandb)
            if last_time_step:
                break

    print('\n ----- Finished all runs -----\n')


def main(args):

    args = boilerplate(args)
    print("***** finish config *****")
    #*************************** Load Datasets ************************#
    meta_train_dataloader, meta_test_dataloader, cl_dataloader = init_dataloaders(args)
    print("**** finish load data ****")
    #*************************** Load Models **************************#
    (metalearner, pretrain_model_loaded) = init_models(args)
    print("**** finish model init ****")
    #********************** Pretrain If Specified **********************#
    pretrain_model_path = args.output + "/" + args.pretrain_model
    print(args.pretrain, pretrain_model_path, os.path.exists(pretrain_model_path))
    if args.pretrain and not pretrain_model_loaded:
        # use a pretrain phase, and do not exist a pretrained model
        print("========start pretrain=======")
        cl_model = pretraining(args, metalearner, meta_train_dataloader, meta_test_dataloader)
        print("**** finish pretrain ****")
    else:
        # no pretrain phase or pretrained model loaded
        cl_model = metalearner
        if not args.pretrain:
            print("**** no pretrain ****")
        else:
            print("**** previous pretrain model loaded ****")


    #******************* Continual Meta Learning ***********************#
    continual_learning(args, cl_model, cl_dataloader)


if __name__ == "__main__":

    args = parse_args()
    #print(args)
    main(args)


