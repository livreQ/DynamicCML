import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions import kl_divergence
from torch.nn import Sigmoid
from torch.nn.functional import relu

import numpy as np
from tqdm import tqdm
from pdb import set_trace
from src.utils.bgd_lib.bgd_optimizer import BGD
import copy
from collections import OrderedDict
from src.utils.utils import update_parameters, tensors_to_device, compute_accuracy
from src.utils.bgd_lib.bgd_optimizer import create_BGD_optimizer

__all__ = ['MAML', 'FOMAML', 'ModularMAML']


class MAML(object):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.

    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).

    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.

    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].

    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.

    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.

    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].

    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.

    device : `torch.device` instance, optional
        The device on which the model is defined.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
           Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)

    .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
           International Conference on Learning Representations (ICLR).
           (https://arxiv.org/abs/1810.09502)
    """
    def __init__(self, model, loss_function, args, optimizer=None):
        self.device = args.device
        self.model = model.to(device=self.device)
        self.optimizer = optimizer
        self.optimizer_cl = None
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.meta_lr)
            if args.bgd_optimizer:
                self.optimizer_cl = create_BGD_optimizer(self.model.to(args.device),
                                                    mean_eta=args.mean_eta,
                                                    std_init=args.std_init,
                                                    mc_iters=args.train_mc_iters)
            else:
                self.optimizer_cl = self.optimizer
        self.step_size = args.step_size
        self.first_order = args.first_order
        ## inner steps
        self.num_adaptation_steps = args.num_steps
        self.scheduler = None
        self.loss_function = loss_function
        self.is_classification_task = args.task == "classification"
        # current task model
        self.current_model = None
        # previous task model
        #self.prev_model = None
        # continual learning related settings
        self.test = args.n_shots_test > 0
        self.cl_strategy = args.cl_strategy
        self.freeze_visual_features = args.freeze_visual_features
        self.no_meta_learning = args.no_cl_meta_learning
        self.best_pretrain_val = None
        self.last_tbd = 0
        self.cl_buffer = []
        self.batch_size = args.batch_size
        self.cl_strategy_thres = args.cl_strategy_thres
        self.cl_tbd_thres = args.cl_tbd_thres

        # self.cl_buffer['inputs'], self.cl_buffer['targets'] = [], []

        # set different learning rate for each parameter
        if args.per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(args.step_size,
                dtype=param.dtype, device=self.device,
                requires_grad=args.learn_step_size)) for (name, param)
                in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(args.step_size, dtype=torch.float32,
                device=self.device, requires_grad=args.learn_step_size)
        # add learning rates into optimizer params
        if (self.optimizer is not None) and args.learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
                if args.per_param_step_size else [self.step_size]})
            if self.scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                    for group in self.optimizer.param_groups])

    def get_outer_loss(self, batch):
        """
            calculate task averaged loss for sgd
        """
        if 'val' not in batch:
            raise RuntimeError('The batch does not contain any validation dataset.')

        _, val_targets = batch['val']
        num_tasks = val_targets.size(0)
        is_classification_task = (not val_targets.dtype.is_floating_point)
        results = { 'num_tasks': num_tasks,
                    'inner_losses': np.zeros((self.num_adaptation_steps, num_tasks), dtype=np.float32),
                    'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
                    'mean_outer_loss': 0.}
        if is_classification_task:
            results.update({'evas_before': np.zeros((num_tasks,), dtype=np.float32),
                            'evas_after': np.zeros((num_tasks,), dtype=np.float32)})

        mean_outer_loss = torch.tensor(0., device=self.device)

        # outer loop: task batches, for online meta, only one task
        for task_id, (train_inputs, train_targets, val_inputs, val_targets) \
                in enumerate(zip(*batch['train'], *batch['val'])):
            # inner task 
            params, adaptation_results = self.inner_update(train_inputs, train_targets)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            if is_classification_task:
                results['evas_before'][task_id] = adaptation_results['eva_before']

            with torch.set_grad_enabled(self.model.training):
                # validation losses
                val_logits = self.model(val_inputs, params=params)
                outer_loss = self.loss_function(val_logits, val_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            if is_classification_task:
                results['evas_after'][task_id] = compute_accuracy(val_logits, val_targets)

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results
    
    def get_outer_loss_bgd(self, inputs, targets, num_of_mc_iters, params=None):
        """
            calculate task losses for bgd
        """
        self.model.zero_grad()
        self.optimizer_cl.zero_grad()
        self.optimizer_cl._init_accumulators()
        outer_loss = []
        acc = 0
        mse = 0
        for mc_iter in range(num_of_mc_iters):
            self.optimizer_cl.randomize_weights()
            self.model.zero_grad()
            self.optimizer_cl.zero_grad()
            if isinstance(self, ModularMAML):
                logits = self.model(inputs, params=self.reset_masks())
            else:
                logits = self.model(inputs, params=params)
            loss = self.loss_function(logits, targets)
            outer_loss.append(loss)
            self.model.zero_grad()
            self.optimizer_cl.zero_grad()
            loss.backward(retain_graph=not self.first_order)
            self.optimizer_cl.aggregate_grads(self.batch_size)
            # self.optimizer.step()
            if self.is_classification_task:
                acc += compute_accuracy(logits, targets)
            else:
                mse += loss
        return acc, mse, outer_loss

    def inner_update(self, inputs, targets):
        """
            base learner adapted on meta-train and meta-val data 
            adapt from meta model (params None)
        """
        params = None
        results = {'inner_losses': np.zeros((self.num_adaptation_steps,), dtype=np.float32)}

        for step in range(self.num_adaptation_steps):
            # if params is None, use model params
            logits = self.model(inputs, params=params)
            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()

            if (step == 0):
                if self.is_classification_task:
                    eva_before = compute_accuracy(logits, targets)
                    results["eva_before"] = eva_before
                else:
                    mse_before = inner_loss
                    results["eva_before"] = mse_before
            self.model.zero_grad()
            # update inner task params with sgd
            params = update_parameters(self.model, inner_loss,
                step_size=self.step_size, params=params,
                first_order=(not self.model.training) or self.first_order,
                freeze_visual_features=self.freeze_visual_features,
                no_meta_learning=self.no_meta_learning)

        return params, results
    
    def adapt_accumulate(self, params, inputs, targets):
        """
            adapt from previous params
        """
        results = {'inner_losses': np.zeros(
            (self.num_adaptation_steps,), dtype=np.float32)}

        for step in range(self.num_adaptation_steps):
            logits = self.model(inputs, params=params)
            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()

            if (step == 0):
                if self.is_classification_task:
                    eva_before = compute_accuracy(logits, targets)
                    results["eva_before"] = eva_before
                else:
                    mse_before = inner_loss
                    results["mse_before"] = mse_before

            self.model.zero_grad()

            params = update_parameters(self.model, inner_loss,
                step_size=self.step_size, params=params,
                first_order=(not self.model.training) or self.first_order,
                freeze_visual_features=self.freeze_visual_features,
                no_meta_learning=self.no_meta_learning)

        return params, results

    def outer_update(self, outer_loss):
        if isinstance(self.optimizer_cl, BGD):
            self.optimizer_cl.step()
        else:
            self.optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                outer_loss.backward()
            self.optimizer.step()

    # meta train
    def meta_train(self, dataloader, max_batches=500, verbose=True, **kwargs):
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.train_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                postfix = {'outer_loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'evas_after' in results:
                    postfix['accuracy'] = '{0:.4f}'.format(np.mean(results['evas_after']))
                if 'inner_losses' in results:
                    postfix['inner_loss'] = '{0:.4f}'.format(np.mean(results['inner_losses']))
                pbar.set_postfix(**postfix)

    def train_iter(self, dataloader, max_batches=500):
        ''' one meta-update '''
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()
        while num_batches < max_batches:
            '''
            for batch in dataloader:
                batch = {'train', 'val'}
                batch['train'][0] = batch_size x m_tr * num_ways x input_dim
                batch['train'][1] = batch_size x m_tr * num_ways x output_dim
                batch['val'][0]  = batch_size x m_va * num_ways x input_dim
                batch['val'][1]  = batch_size x m_va * num_ways x output_dim
            '''
            for batch in dataloader:# task batch size is determined by dataloader
            #for i, batch in enumerate(dataloader):
                if num_batches >= max_batches:
                    break
                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)
                self.optimizer.zero_grad()
                batch = tensors_to_device(batch, device=self.device)
                outer_loss, results = self.get_outer_loss(batch)
                yield results
                outer_loss.backward()
                self.optimizer.step()
                num_batches += 1

    # meta test
    def meta_test(self, dataloader, max_batches=500, verbose=True, epoch=0, **kwargs):
        mean_outer_loss, mean_inner_loss, mean_accuracy, mean_accuracy_before, count = 0., 0., 0., 0., 0
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'evas_before' in results:
                    mean_accuracy_before += (np.mean(results['evas_before'])
                        - mean_accuracy_before) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy_before)
                if 'evas_after' in results:
                    mean_accuracy += (np.mean(results['evas_after'])
                        - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                if 'inner_losses' in results:
                    mean_inner_loss += (np.mean(results['inner_losses'])
                        - mean_inner_loss) / count
                    postfix['inner_loss'] = '{0:.4f}'.format(mean_inner_loss)
                pbar.set_postfix(**postfix)

        results = {
            'mean_outer_loss': mean_outer_loss,
            'evas_before': mean_accuracy_before,
            'evas_after': mean_accuracy,
            'mean_inner_loss': mean_inner_loss,
        }

        return results

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break
                batch = tensors_to_device(batch, device=self.device)
                _, results = self.get_outer_loss(batch)
                yield results
                num_batches += 1

    # continual learning
    # def observe(self, batch):
        # if self.cl_strategy == 'never_retrain':
        #     self.model.eval()
        # else:
        #     self.model.train()
        
        # # one task
        # data, task_switch , env_switch, env = batch
        # inputs, targets = data['train']
        # # for now we are doing one task at a time
        # assert inputs.shape[0] == 1
        # assert self.optimizer_cl != None, 'Set optimizer_cl'
        # # mc sampling for bgd optimizer
        # self.batch_size = inputs.shape[1]
        # num_of_mc_iters = 1
        # #set_trace()
        # if hasattr(self.optimizer_cl, "get_mc_iters"):
        #     num_of_mc_iters = self.optimizer_cl.get_mc_iters()
        # inputs, targets  = inputs[0], targets[0]
        # results = {'inner_losses': np.zeros((self.num_adaptation_steps,), dtype=np.float32),
        #            'outer_loss': 0.,
        #            'task_boundary':0.,
        #            'env_switch':0.,}
        # if self.current_model is None:
        #     print("current_model is none")
        #     self.current_model, results = self.inner_update(inputs, targets)
        #     self.last_env = env[0]
        #     results['eva'] = results['eva_before']
        #     results['env_switch'] = 0
        #     results['task_boundary'] = 0
        #     if self.test:
        #         print("==========separate test==============")
        #         test_inputs, test_targets = data['val']
        #         test_inputs, test_targets = test_inputs[0], test_targets[0]
        #         with torch.no_grad():
        #             test_logits = self.model(test_inputs, params=self.current_model)
        #             # outer_loss = self.loss_function(logits, test_targets).item()
        #             if self.is_classification_task:
        #                 results['eva_test'] = compute_accuracy(test_logits, test_targets)
        #             else:
        #                 results["eva_test"] = F.mse_loss(test_logits, test_targets)
        #     return results
        # #self.prev_model = copy.deepcopy(self.current_model)
        # #self.prev_model = self.current_model
        # ## inplace model params are not changed
        # self.current_model, _ = self.inner_update(inputs, targets)
        # if self.test:
        #     print("==========separate test==============")
        #     test_inputs, test_targets = data['val']
        #     test_inputs, test_targets = test_inputs[0], test_targets[0]
        #     with torch.no_grad():
        #         test_logits = self.model(test_inputs, self.current_model)
        #         # outer_loss = self.loss_function(logits, test_targets).item()
        #         if self.is_classification_task:
        #             results['eva_test'] = compute_accuracy(test_logits, test_targets)
        #         else:
        #             results["eva_test"] = F.mse_loss(test_logits, test_targets)
                    
        # with torch.no_grad():
        #     logits = self.model(inputs, self.current_model)
        #     current_outer_loss = self.loss_function(logits, targets).detach().item()
        #     current_acc = compute_accuracy(logits, targets)
        
        # #assert self.prev_model != self.current_model
        # ## try the prev task model params on the incoming data:
        # with torch.set_grad_enabled(self.model.training):
        #     if isinstance(self.optimizer_cl, BGD):
        #         ## using BGD:
        #         acc, mse, outer_loss = self.get_outer_loss_bgd(inputs, targets, num_of_mc_iters)
        #         if self.is_classification_task:
        #             results['eva'] = acc / num_of_mc_iters            
        #         else:
        #             results["eva"]  = mse / num_of_mc_iters
        #         results['outer_loss'] = torch.mean(torch.tensor(outer_loss)).item()
        #     else:
        #         ## using SGD
        #         # previous task model/meta model 
        #         logits = self.model(inputs)#, params=self.prev_model)
        #         outer_loss = self.loss_function(logits, targets)
        #         with torch.no_grad():
        #             results['outer_loss'] = outer_loss.item()
        #             if self.is_classification_task:
        #                 results['eva'] = compute_accuracy(logits, targets)
        #             else:
        #                 results["eva"] = F.mse_loss(logits, targets)


        # #----------------- CL strategies ------------------#

        # # Note: this is the C-MAML algo w/o the prolonged adaptation phase (PAP)
        # # and with the discrete version of the update modulation (UM)
        
        # #flag of task boundary detection
        # tbd = 0
        # if self.cl_tbd_thres >= 0 and self.cl_tbd_thres < 100 :
        #     ## task not switched 
        #     if self.cl_strategy=='acc':
        #         if current_acc >= results['eva'] + self.cl_tbd_thres:
        #             tbd = 1
        #     elif self.cl_strategy=='loss':
        #         if current_outer_loss + self.cl_tbd_thres <= results['outer_loss']:
        #             tbd = 1
        # ood = 1
        # # use accuracy/loss difference to decide if an environment shift exists
        # if self.cl_strategy in ['loss', 'acc']:
        #     if self.cl_strategy=='acc':
        #         if results['eva'] >= self.cl_strategy_thres:
        #             ood = 0
        #     elif 'loss' in str(self.cl_strategy):
        #         if results['outer_loss'] <= self.cl_strategy_thres:
        #             ood = 0
       
        # # retrain and env shift exists and task boundary detected, update meta model
        # if self.cl_strategy != 'never_retrain' and not tbd and ood:
        #     if self.cl_strategy != 'loss_smooth':
        #         self.outer_update(outer_loss)
        #     else:
        #         smoothing_weight = (1 - torch.exp(-self.cl_strategy_thres * outer_loss.detach()))
        #         self.outer_update(smoothing_weight * outer_loss)
        #         # print(smoothing_weight)
        # results['task_boundary'] = tbd
        # results['env_switch'] = ood
        # return results
    
    def observe(self, batch):
        if self.cl_strategy == 'never_retrain':
            self.model.eval()
        else:
            self.model.train()
        
        #inputs, targets, _ , _ = batch
        # one task
        data, task_switch , env_switch, env = batch
        inputs, targets = data['train']
        # for now we are doing one task at a time
        assert inputs.shape[0] == 1
        assert self.optimizer_cl != None, 'Set optimizer_cl'
        # mc sampling for bgd optimizer
        self.batch_size = inputs.shape[1]
        num_of_mc_iters = 1
        #set_trace()
        if hasattr(self.optimizer_cl, "get_mc_iters"):
            num_of_mc_iters = self.optimizer_cl.get_mc_iters()
        inputs, targets  = inputs[0], targets[0]
        results = {'inner_losses': np.zeros((self.num_adaptation_steps,), dtype=np.float32),
                   'outer_loss': 0.,
                   'task_boundary':0.,
                   'env_switch':0.,}
        # if self.is_classification_task:
        #     results.update({'eva_before': 0.,'eva': 0., 'eva_test':0})
        # else:
        #     results.update({"mse_before": 0., "mse_after": 0.,})
        if self.current_model is None:
            # update random initialization to first updated model 
            self.current_model, results = self.inner_update(inputs, targets)
            self.last_env = env[0]
            results['eva'] = results['eva_before']
            results['env_switch'] = 0
            results['task_boundary'] = 0
            if self.test:
                #print("==========separate test==============")
                test_inputs, test_targets = data['val']
                test_inputs, test_targets = test_inputs[0], test_targets[0]
                with torch.no_grad():
                    test_logits = self.model(test_inputs, params=self.current_model)
                    # outer_loss = self.loss_function(logits, test_targets).item()
                    if self.is_classification_task:
                        results['eva_test'] = compute_accuracy(test_logits, test_targets)
                    else:
                        results["eva_test"] = F.mse_loss(test_logits, test_targets)
            return results

        ## try the prev meta model params on the incoming data:
        with torch.set_grad_enabled(self.model.training):
            if isinstance(self.optimizer_cl, BGD):
                ## using BGD:
                acc, mse, outer_loss  = self.get_outer_loss_bgd(inputs, targets, num_of_mc_iters)
                if self.is_classification_task:
                    results['eva'] = acc / num_of_mc_iters
                else:
                    results["eva"]  = mse / num_of_mc_iters
                results['outer_loss'] = torch.mean(torch.tensor(outer_loss)).item()
            else:
                ## using SGD
                # previous task model/meta model 
                logits = self.model(inputs, params=self.current_model)
                outer_loss = self.loss_function(logits, targets)
                results['outer_loss'] = outer_loss.item()
                if self.is_classification_task:
                    results['eva'] = compute_accuracy(logits, targets)
                else:
                    results["eva"] = F.mse_loss(logits, targets)

        ## prediction is done and you can now use the labels
        
        self.current_model, _ = self.inner_update(inputs, targets)

        if self.test:
            #print("==========separate test==============")
            test_inputs, test_targets = data['val']
            test_inputs, test_targets = test_inputs[0], test_targets[0]
            with torch.no_grad():
                test_logits = self.model(test_inputs, params=self.current_model)
                # outer_loss = self.loss_function(logits, test_targets).item()
                if self.is_classification_task:
                    results['eva_test'] = compute_accuracy(test_logits, test_targets)
                else:
                    results["eva_test"] = F.mse_loss(test_logits, test_targets)
        #----------------- CL strategies ------------------#

        # Note: this is the C-MAML algo w/o the prolonged adaptation phase (PAP)
        # and with the discrete version of the update modulation (UM)
        
        #flag of task boundary detection
        tbd = 0
        if self.cl_tbd_thres >= 0 and self.cl_tbd_thres < 100 :
            with torch.no_grad():
                logits = self.model(inputs, params=self.current_model)
                current_outer_loss = self.loss_function(logits, targets).detach().item()
                current_acc = compute_accuracy(logits, targets)

            ## adapt from previous model much better than adapt from meta-model, task not switched 
            if self.cl_strategy=='acc':
                # results -> update from previous model, current_acc -> update from meta-model 
                if current_acc >= results['eva'] + self.cl_tbd_thres:
                    tbd = 1
            elif self.cl_strategy=='loss':
                if current_outer_loss + self.cl_tbd_thres <= results['outer_loss']:
                    tbd = 1

        if self.cl_tbd_thres == -2:
            """
               always switch task
            """
            tbd = 1

        ood = 1
        # use accuracy/loss difference to decide if an environment shift exists
        if self.cl_strategy in ['loss', 'acc']:
            if self.cl_strategy=='acc':
                if results['eva'] >= self.cl_strategy_thres:
                    ood = 0
            elif 'loss' in str(self.cl_strategy):
                if results['outer_loss'] <= self.cl_strategy_thres:
                    ood = 0

        # ood domain, different from pretrained (meta) domain
        # retrain and env shift exists but task boundary not detected, update meta model
        if self.cl_strategy != 'never_retrain' and not tbd and ood:
            logits = self.model(inputs)#, params=self.current_model)
            outer_loss = self.loss_function(logits, targets)
            if self.cl_strategy != 'loss_smooth':
                self.outer_update(outer_loss)
            else:
                smoothing_weight = (1 - torch.exp(-self.cl_strategy_thres * outer_loss.detach()))
                self.outer_update(smoothing_weight * outer_loss)
                # print(smoothing_weight)
        results['task_boundary'] = tbd
        results['env_switch'] = ood
        return results


    def observe_accumulate(self, batch):
        if self.cl_strategy == 'never_retrain':
            self.model.eval()
        else:
            self.model.train()

        #inputs, targets, _ , _ = batch
        #inputs, targets, task_switch , env = batch
        data, task_switch , env_switch, env = batch
        inputs, targets = data['train']
        # for now we are doing one task at a time
        assert inputs.shape[0] == 1
        assert self.optimizer_cl != None, 'Set optimizer_cl'
        # mc sampling for bgd optimizer
        self.batch_size = inputs.shape[1]
        num_of_mc_iters = 1
        #set_trace()
        if hasattr(self.optimizer_cl, "get_mc_iters"):
            num_of_mc_iters = self.optimizer_cl.get_mc_iters()
        inputs, targets  = inputs[0], targets[0]

        results = {}
        # {'inner_losses': np.zeros((self.num_adaptation_steps,), dtype=np.float32), 'outer_loss': 0., 'tbd':0.,}
        # if self.is_classification_task:
        #     results.update({'eva_before': 0., 'eva': 0.})
        # else:
        #     results.update({"mse_before": 0., "mse_after": 0.,})

        if self.current_model is None:
            self.current_model, results = self.inner_update(inputs, targets)
            self.last_env = env[0]
            self.cl_buffer.append(batch)
            results['eva'] = results['eva_before']
            results['env_switch'] = 0
            results['task_boundary'] = 0
            if self.test:
                print("==========separate test==============")
                test_inputs, test_targets = data['val']
                test_inputs, test_targets = test_inputs[0], test_targets[0]
                with torch.no_grad():
                    logits = self.model(test_inputs, params=self.current_model)
                    # outer_loss = self.loss_function(logits, test_targets).item()
                    if self.is_classification_task:
                        results['eva_test'] = compute_accuracy(logits, test_targets)
                    else:
                        results["eva_test"] = F.mse_loss(logits, test_targets)
            return results

        ## try the prev model on the incoming data:
        with torch.set_grad_enabled(self.model.training):
            if isinstance(self.optimizer_cl, BGD):
                ## using BGD:
                acc, mse, outer_loss  = self.get_outer_loss_bgd(inputs, targets, num_of_mc_iters)
                if self.is_classification_task:
                    results['eva'] = acc / num_of_mc_iters
                else:
                    results["eva"]  = mse / num_of_mc_iters
                results['outer_loss'] = torch.mean(torch.tensor(outer_loss)).item()
            else:
                ## using SGD
                # evaluate current data on previous model
                logits = self.model(inputs, params=self.current_model)
                outer_loss = self.loss_function(logits, targets)
                results['outer_loss'] = outer_loss.item()
                if self.is_classification_task:
                    results['eva'] = compute_accuracy(logits, targets)
                else:
                    results["eva"] = F.mse_loss(logits, targets)

        ## prediction is done and you can now use the labels

        self.model.eval()
        if self.last_tbd:
            # print('reinit the model') 
            self.current_model, _ = self.inner_update(inputs, targets)
        else:   
            self.current_model, _ = self.adapt_accumulate(self.current_model, inputs, targets)
        
        # use another data to test performance, fast adaptation
        if self.test:
            test_inputs, test_targets = data['val']
            test_inputs, test_targets = test_inputs[0], test_targets[0]
            with torch.no_grad():
                logits = self.model(test_inputs, params=self.current_model)
                #outer_loss = self.loss_function(logits, test_targets).item()
                if self.is_classification_task:
                    results['eva_test'] = compute_accuracy(logits, test_targets)
                else:
                    results["eva_test"] = F.mse_loss(logits, test_targets)
        #----------------- CL strategies ------------------#

        tbd = 0
        if self.cl_tbd_thres >= 0 and self.cl_tbd_thres < 100 :

            with torch.no_grad():
                logits = self.model(inputs, params=self.current_model)
                current_outer_loss = self.loss_function(logits, targets).item()
                current_acc = compute_accuracy(logits, targets)

            ## if task switched, than inner and outer loop have a missmatch!
            if self.cl_strategy=='acc':
                if current_acc >= results['eva'] + self.cl_tbd_thres:
                    tbd = 1
            elif 'loss' in str(self.cl_strategy):
                if current_outer_loss + self.cl_tbd_thres <= results['outer_loss']:
                    tbd = 1

        if self.cl_tbd_thres == -2:
            """
                always switch task
            """
            tbd = 1

        ood = 1
        if self.cl_strategy in ['loss', 'acc']:

            if self.cl_strategy=='acc':
                if results['eva'] >= self.cl_strategy_thres:
                    ood = 0

            elif self.cl_strategy=='loss':
                if results['outer_loss'] <= self.cl_strategy_thres:
                    ood = 0

        if (tbd and len(self.cl_buffer)>0) or (len(self.cl_buffer)>2*self.batch_size):
            
            #Note: we enter here when a task boundary as been detected and that it's time to update \phi
            # or if the the buffer is close to full. Then we also update \phi and restart the buffer
            
            batch = self.make_batch()
            self.model.train()
            outer_loss, _ = self.get_outer_loss(batch)
            if self.cl_strategy != 'loss_smooth':
                self.outer_update(outer_loss)
            else:
                smoothing_weight = (1-torch.exp(-self.cl_strategy_thres * outer_loss.detach()))
                self.outer_update(smoothing_weight * outer_loss)
                # print(smoothing_weight)
            self.model.eval()

            ## restart buffer
            self.cl_buffer = []
            # print('updating model and restarting buffer')
        else:
            self.cl_buffer.append(batch)
        #--------------------------------------------------#
        self.last_tbd = tbd
        results['task_boundary'] = tbd
        results['env_switch'] = ood
        return results

    def make_batch(self):
        if len(self.cl_buffer)==1:
            ## oups
            self.cl_buffer.append(self.cl_buffer[0])
        idx = int(np.ceil(len(self.cl_buffer)/2))
        batch = {}
        train_x = []
        train_y = []
        for i in range(idx):
            if i==0:
                train_x = self.cl_buffer[i][0]['train'][0]
                train_y = self.cl_buffer[i][0]['train'][1]
            else:
                train_x = torch.cat([train_x, self.cl_buffer[i][0]['train'][0]])
                train_y = torch.cat([train_y, self.cl_buffer[i][0]['train'][1]])
        batch['train'] = [train_x, train_y]
        val_x = []
        val_y = []
        for i in range(idx, len(self.cl_buffer)):
            if i==idx:
                val_x = self.cl_buffer[i][0]['train'][0]
                val_y = self.cl_buffer[i][0]['train'][1]
            else:
                val_x = torch.cat([val_x, self.cl_buffer[i][0]['train'][0]])
                val_y = torch.cat([val_y, self.cl_buffer[i][0]['train'][1]])
        batch['val'] = [val_x, val_y]
        return batch

    
    
class FOMAML(MAML):
    def __init__(self, model, optimizer=None, step_size=0.1,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        super(FOMAML, self).__init__(model, optimizer=optimizer, first_order=True,
            step_size=step_size, learn_step_size=learn_step_size,
            per_param_step_size=per_param_step_size,
            num_adaptation_steps=num_adaptation_steps, scheduler=scheduler,
            loss_function=loss_function, device=device)


class ModularMAML(MAML):
    def __init__(self, model, loss_function, args, wandb=None, optimizer=None):
        super(ModularMAML, self).__init__(model, loss_function, args, optimizer)

        assert (args.kl_reg<=0) or args.mask_activation=='sigmoid'

        self.mask_activation = args.mask_activation
        self.modularity = args.modularity
        self.l1_reg = args.l1_reg
        self.kl_reg = args.kl_reg
        self.bern_prior = args.bern_prior
        self.masks_init = args.masks_init
        self.hard_masks = args.hard_masks
        self.wandb = wandb
        self.current_mask_stats = None

        self.weight_pruning = OrderedDict(self.model.meta_named_parameters())
        self.weight_total = OrderedDict(self.model.meta_named_parameters())
        self.reset_weight_pruning()

        # count total number of params
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.tot_params = sum([np.prod(p.size()) for p in model_parameters])

    def reset_weight_pruning(self):
        if self.modularity == 'param_wise':
            for (name, _) in self.weight_pruning.items():
                if 'classifier' in name:
                    continue
                self.weight_pruning[name] = torch.autograd.Variable(
                    torch.zeros_like(self.weight_pruning[name]), requires_grad=False).type(torch.int)
                self.weight_total[name] = torch.autograd.Variable(
                     torch.zeros_like(self.weight_total[name]), requires_grad=False).type(torch.int)

    def apply_non_linearity(self, masks_logits):
        if self.mask_activation in [None, 'None']:
            if self.hard_masks:
                return torch.clamp(masks_logits, 1e-8, 1-1e-8)
            else:
                return masks_logits
        elif self.mask_activation == 'sigmoid':
            return Sigmoid()(masks_logits)
        elif self.mask_activation == 'ReLU':
            if self.hard_masks:
                return torch.clamp(masks_logits, 1e-8, 1-1e-8)
            else:
                return relu(masks_logits)
        elif self.mask_activation == 'hardsrink':
            raise Exception('doesnt work yet')
            return torch.nn.Hardshrink()(masks_logits)

    def init_params(self):

        params = OrderedDict(self.model.meta_named_parameters())
        params_masked = OrderedDict(self.model.meta_named_parameters())
        masks_logits = OrderedDict(self.model.meta_named_parameters())
        masks = OrderedDict(self.model.meta_named_parameters())

        #TODO(learn the initial value)
        if self.modularity=='param_wise':
            for (name, _) in masks_logits.items():
                if 'classifier' in name:
                    continue
                else:
                    masks_logits[name] = torch.autograd.Variable(torch.ones_like(masks_logits[name])*
                            self.masks_init, requires_grad=True)
                    masks[name] = torch.autograd.Variable(torch.zeros_like(masks[name]),
                            requires_grad=True)

        return params, params_masked, masks_logits, masks

    def apply_masks(self, params, params_masked, masks_logits, masks, regularize=False, evaluate=False):

        l1_reg, kl_reg = 0, 0

        for (name, _) in masks_logits.items():

            if 'classifier' in name:
                # we are not pruning the classifier:
                params_masked[name] = masks_logits[name]

            else:
                masks[name] = self.apply_non_linearity(masks_logits[name])

                # we could to hard mask this way, but less interpretable
                #applied_masks = masks[name] * (masks[name].detach()>self.masks_thres).float()
                if self.hard_masks:
                    applied_masks = Bernoulli(probs=masks[name]).sample()
                    applied_masks = (masks[name] + applied_masks).detach() - masks[name]
                else:
                    applied_masks = masks[name]

                if self.modularity=='param_wise':
                    params_masked[name] = params[name] * applied_masks

                if regularize:
                    if self.l1_reg>0:
                        l1_reg += self.l1_reg * torch.sum(torch.abs(masks[name]))

                    if self.kl_reg>0:
                        # this will only work if masks = sigmoid(masks_logits)
                        bern_masks = Bernoulli(probs=masks[name])
                        bern_prior = Bernoulli(probs=torch.ones_like(masks[name])*self.bern_prior)
                        kl_reg += self.kl_reg * \
                                torch.distributions.kl_divergence(bern_masks, bern_prior).sum()

                # count the number of pruned neurons
                if evaluate:
                    self.weight_pruning[name] += (applied_masks==0).type(torch.int)
                    self.weight_total[name] += torch.ones_like(applied_masks).type(torch.int)

        if regularize:
            reg = l1_reg + kl_reg
            return params_masked, masks_logits, reg
        else:
            return params_masked, masks_logits

    def inner_update(self, inputs, targets):

        results = {'inner_losses': np.zeros(
            (self.num_adaptation_steps,), dtype=np.float32)}

        params, params_masked, masks_logits, masks = self.init_params()

        for step in range(self.num_adaptation_steps):

            params_masked, masks_logits, reg = self.apply_masks(params, params_masked, masks_logits,
                    masks, regularize=True)

            logits = self.model(inputs, params=params_masked)
            inner_loss = self.loss_function(logits, targets) + reg

            results['inner_losses'][step] = inner_loss.item()

            if (step == 0) and self.is_classification_task:
                results['eva_before'] = compute_accuracy(logits, targets)

            self.model.zero_grad()

            masks_logits = update_parameters(self.model, inner_loss,
                step_size=self.step_size, params=masks_logits,
                first_order=(not self.model.training) or self.first_order,
                freeze_visual_features = self.freeze_visual_features,
                no_meta_learning=self.no_meta_learning)

        self.current_mask_stats = masks_logits
        # final masking
        params_masked, _ = self.apply_masks(params, params_masked, masks_logits, masks,
                    regularize=False, evaluate=(not self.model.training))

        return params_masked, results

    def sparsity_monitoring(self, epoch):
        tot_sparsity, tot_dead = [], []
        params = OrderedDict(self.model.meta_named_parameters())
        for (name, _) in self.weight_pruning.items():
            if 'classifier' in name:
                continue
            sparsity = self.weight_pruning[name].float() / self.weight_total[name].float()
            spartity = sparsity.cpu().numpy()
            sparsity_mean = sparsity.mean()
            sparsity_std = sparsity.std()
            sparsity = sparsity.flatten().tolist()
            multiplier=1
            tot_sparsity += sparsity * multiplier
            dead = self.weight_pruning[name] == self.weight_total[name]
            dead = dead.type(torch.float).cpu().numpy()
            dead_mean = dead.mean()
            dead_std = dead.std()
            dead = dead.flatten().tolist()
            tot_dead += dead * multiplier
            print(name + ' : sparse={0:.3f} +\- {1:.3f} \t dead={2:.3f} +/- {3:.3f}'.format(
                sparsity_mean, sparsity_std, dead_mean, dead_std))
            if self.wandb is not None:
                self.wandb.log({name+'_sparse_mean':sparsity_mean}, step=epoch)
                self.wandb.log({name+'_sparse_std':sparsity_std}, step=epoch)
                self.wandb.log({name+'_dead_mean':dead_mean}, step=epoch)
                self.wandb.log({name+'_dead_std':dead_std}, step=epoch)

        tot_sparsity_mean = np.array(tot_sparsity).mean()
        tot_sparsity_std = np.array(tot_sparsity).std()
        tot_dead_mean = np.array(tot_dead).mean()
        tot_dead_std = np.array(tot_dead).std()
        print('Total : sparse={0:.3f} +\- {1:.3f} \t dead={2:.3f} +/- {3:.3f}'.format(
                tot_sparsity_mean, tot_sparsity_std, tot_dead_mean, tot_dead_std))
        if self.wandb is not None:
            self.wandb.log({'tot_sparsity_mean':tot_sparsity_mean}, step=epoch)
            self.wandb.log({'tot_sparsity_std':tot_sparsity_std}, step=epoch)
            self.wandb.log({'tot_dead_mean':tot_dead_mean}, step=epoch)
            self.wandb.log({'tot_dead_std':tot_dead_std}, step=epoch)

        self.reset_weight_pruning()

    def evaluate(self, dataloader, max_batches=500, verbose=True, epoch=0, **kwargs):
        mean_outer_loss, mean_inner_loss, mean_accuracy, mean_accuracy_before, count = 0., 0., 0., 0., 0
        self.reset_weight_pruning()

        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                # one task batch
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'evas_before' in results:
                    mean_accuracy_before += (np.mean(results['evas_before'])
                        - mean_accuracy_before) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                if 'evas_after' in results:
                    mean_accuracy += (np.mean(results['evas_after'])
                        - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                if 'inner_losses' in results:
                    mean_inner_loss += (np.mean(results['inner_losses'])
                        - mean_inner_loss) / count
                    postfix['inner_loss'] = '{0:.4f}'.format(mean_inner_loss)
                pbar.set_postfix(**postfix)

        self.sparsity_monitoring(epoch)

        results = {
            'mean_outer_loss': mean_outer_loss,
            'evas_before': mean_accuracy_before,
            'evas_after': mean_accuracy,
            'mean_inner_loss': mean_inner_loss
        }

        return results

    def reset_masks(self, params=None):
        if params is None:
            params = OrderedDict(self.model.meta_named_parameters())
        params_masked = OrderedDict(self.model.meta_named_parameters())
        masks = OrderedDict(self.model.meta_named_parameters())
        masks_logits = OrderedDict(self.model.meta_named_parameters())

        params_masked, _ = self.apply_masks(params, params_masked, masks_logits, masks,
                                           regularize=False, evaluate=(not self.model.training))
        return params_masked


