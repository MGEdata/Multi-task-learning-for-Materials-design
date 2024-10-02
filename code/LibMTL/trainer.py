import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss
import matplotlib.pyplot as plt
import shap

task_features = {
        'solvus': 24,
        'solidus': 24,
        'liquidus': 24,
        'density': 24,
        'size': 14,
        'phase_class': 11,
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_nested_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_nested_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def shap_parse(model,x_test):

    # 使用 SHAP 进行解释
    explainer = shap.Explainer(model, x_test.numpy())
    shap_values = explainer.shap_values(x_test.numpy())

    # 可视化 SHAP 值
    shap.summary_plot(shap_values, features=x_test.numpy(), feature_names=[f"Feature {i}" for i in range(24)],
                      show=False)

    # 添加标题和保存图像
    plt.title('SHAP Values for Features')
    plt.savefig(r'E:\desktop\shap\shap_values_plot.png')
    plt.show()



def mtl_r2(predict_results_train,true_y):
    task_true = dict()
    task_predict = dict()
    data_i = 0
    for batch_data in predict_results_train:
        if list(batch_data.keys())[0] not in task_true.keys():
            if type(torch.squeeze(batch_data[list(batch_data.keys())[0]]).tolist()) == list:
                task_predict[list(batch_data.keys())[0]] = list()
                task_predict[list(batch_data.keys())[0]].extend(
                    torch.squeeze(batch_data[list(batch_data.keys())[0]]).tolist())
                task_true[list(batch_data.keys())[0]] = list()
                task_true[list(batch_data.keys())[0]].extend(true_y[data_i])
            elif type(torch.squeeze(batch_data[list(batch_data.keys())[0]]).tolist()) == float:
                task_predict[list(batch_data.keys())[0]] = list()
                task_predict[list(batch_data.keys())[0]].append(
                    torch.squeeze(batch_data[list(batch_data.keys())[0]]).tolist())
                task_true[list(batch_data.keys())[0]] = list()
                task_true[list(batch_data.keys())[0]].extend(true_y[data_i])
        else:
            # print(list(batch_data.keys()))
            # print(task_predict[list(batch_data.keys())[0]])
            # print(torch.squeeze(batch_data[list(batch_data.keys())[0]]).tolist())
            if type(torch.squeeze(batch_data[list(batch_data.keys())[0]]).tolist()) == list:
                task_predict[list(batch_data.keys())[0]].extend(
                    torch.squeeze(batch_data[list(batch_data.keys())[0]]).tolist())
                task_true[list(batch_data.keys())[0]].extend(true_y[data_i])
            elif type(torch.squeeze(batch_data[list(batch_data.keys())[0]]).tolist()) == float:
                task_predict[list(batch_data.keys())[0]].append(
                    torch.squeeze(batch_data[list(batch_data.keys())[0]]).tolist())
                task_true[list(batch_data.keys())[0]].extend(true_y[data_i])
        data_i += 1
    task_score = dict()
    for name,info in task_true.items():
        # 将列表转换为NumPy数组
        true = np.array(task_true[name])
        predict = np.array(task_predict[name])
        if name != "phase_class":
            # 计算MSE误差
            value = r2_score(true, predict)
            loss = ((predict - true) ** 2).mean()
        else:
            predicted_labels = (predict >= 0.5).astype(int)
            value = accuracy_score(true, predicted_labels)
            criterion = nn.BCELoss()
            true = torch.tensor(true, dtype=torch.float32)
            predict = torch.tensor(predict, dtype=torch.float32)
            # loss = criterion(predict, true)
            loss = log_loss(true, predict)
        task_score[name] = (value,loss.item())
    return task_score

class Trainer(nn.Module):
    r'''A Multi-Task Learning Trainer.

    This is a unified and extensible training framework for multi-task learning.

    Args:
        task_dict (dict): A dictionary of name-information pairs of type (:class:`str`, :class:`dict`). \
                            The sub-dictionary for each task has four entries whose keywords are named **metrics**, \
                            **metrics_fn**, **loss_fn**, **weight** and each of them corresponds to a :class:`list`.
                            The list of **metrics** has ``m`` strings, repersenting the name of ``m`` metrics \
                            for this task. The list of **metrics_fn** has two elements, i.e., the updating and score \
                            functions, meaning how to update thoes objectives in the training process and obtain the final \
                            scores, respectively. The list of **loss_fn** has ``m`` loss functions corresponding to each \
                            metric. The list of **weight** has ``m`` binary integers corresponding to each \
                            metric, where ``1`` means the higher the score is, the better the performance, \
                            ``0`` means the opposite.
        weighting (class): A weighting strategy class based on :class:`LibMTL.weighting.abstract_weighting.AbsWeighting`.
        architecture (class): An architecture class based on :class:`LibMTL.architecture.abstract_arch.AbsArchitecture`.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``.
        optim_param (dict): A dictionary of configurations for the optimizier.
        scheduler_param (dict): A dictionary of configurations for learning rate scheduler. \
                                 Set it to ``None`` if you do not use a learning rate scheduler.
        kwargs (dict): A dictionary of hyperparameters of weighting and architecture methods.

    .. note::
            It is recommended to use :func:`LibMTL.config.prepare_args` to return the dictionaries of ``optim_param``, \
            ``scheduler_param``, and ``kwargs``.

    Examples::

        import torch.nn as nn
        from LibMTL import Trainer
        from LibMTL.loss import CE_loss_fn
        from LibMTL.metrics import acc_update_fun, acc_score_fun
        from LibMTL.weighting import EW
        from LibMTL.architecture import HPS
        from LibMTL.model import ResNet18
        from LibMTL.config import prepare_args

        task_dict = {'A': {'metrics': ['Acc'],
                           'metrics_fn': [acc_update_fun, acc_score_fun],
                           'loss_fn': [CE_loss_fn],
                           'weight': [1]}}

        decoders = {'A': nn.Linear(512, 31)}

        # You can use command-line arguments and return configurations by ``prepare_args``.
        # kwargs, optim_param, scheduler_param = prepare_args(params)
        optim_param = {'optim': 'adam', 'lr': 1e-3, 'weight_decay': 1e-4}
        scheduler_param = {'scheduler': 'step'}
        kwargs = {'weight_args': {}, 'arch_args': {}}

        trainer = Trainer(task_dict=task_dict,
                          weighting=EW,
                          architecture=HPS,
                          encoder_class=ResNet18,
                          decoders=decoders,
                          rep_grad=False,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)

    '''

    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, rep_grad, multi_input,
                 optim_param, scheduler_param,
                 save_path=None, load_path=None, **kwargs):
        super(Trainer, self).__init__()

        self.device = device
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param
        self.save_path = save_path
        self.load_path = load_path

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param)

        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)

    def _prepare_model(self, weighting, architecture, encoder_class, decoders):

        class MTLmodel(architecture, weighting):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input,
                                               device, **kwargs)
                self.init_param()

        self.model = MTLmodel(task_name=self.task_name,
                              encoder_class=encoder_class,
                              decoders=decoders,
                              rep_grad=self.rep_grad,
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                self.load_path = os.path.join(self.load_path, 'best.pt')
            self.model.load_state_dict(torch.load(self.load_path), strict=False)
            print('Load Model from - {}'.format(self.load_path))
        count_parameters(self.model)

    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adagrad': torch.optim.Adagrad,
            'rmsprop': torch.optim.RMSprop,
        }
        scheduler_dict = {
            'exp': torch.optim.lr_scheduler.ExponentialLR,
            'step': torch.optim.lr_scheduler.StepLR,
            'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
            'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
        }
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _process_data(self, loader):

        # if self.multi_input:
        #     for task in self.task_name:
        #         data, label = next(iter(loader))
        #         data = data.to(self.device, non_blocking=True)
        #         # label[task] = label[task].to(self.device, non_blocking=True)
        #         label = label.to(self.device, non_blocking=True)
        # else:
        data, data_plus, label = next(iter(loader))
        data = data.to(self.device, non_blocking=True)
        data_plus = data_plus.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)
        return data, data_plus, label

    def process_preds(self, preds, task_name=None):
        r'''The processing of prediction for each task.

        - The default is no processing. If necessary, you can rewrite this function.
        - If ``multi_input`` is ``True``, ``task_name`` is valid and ``preds`` with type :class:`torch.Tensor` is the prediction of this task.
        - otherwise, ``task_name`` is invalid and ``preds`` is a :class:`dict` of name-prediction pairs of all tasks.

        Args:
            preds (dict or torch.Tensor): The prediction of ``task_name`` or all tasks.
            task_name (str): The string of task name.
        '''
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        gts = gts.unsqueeze(1)
        # print(gts.size())
        if self.multi_input:
            train_losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                # train_losses[tn] = self.meter.losses[task]._update_loss(preds[task].to(torch.float32), gts[task].to(torch.float32))
                if task in preds.keys():

                    # if task == "phase_class":
                    #     train_losses[tn] = self.meter.losses[task]._update_loss(preds[task].to(torch.float32),
                    #                                                             gts.to(torch.float32))
                    #
                    # else:
                    train_losses[tn] = self.meter.losses[task]._update_loss(preds[task].to(torch.float32),
                                                                                gts.to(torch.float32))

        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)
        return train_losses

    def _prepare_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                if task in dataloaders.keys():
                    loader[task] = dataloaders[task]
                    batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def train(self, train_dataloaders, data_train,data_test,data_all, epochs, fig_id, weights,
              val_dataloaders=None, return_weight=False,params=[]):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''

        all_predicts = list()
        all_trues = list()

        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch

        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        self.task_name = data_train.task_names
        train_all = dict()
        vals_all = dict()
        epoch2score = list()
        task_commons = None
        print(self.task_name)
        for tn, task in enumerate(self.task_name):
            train_all[task] = []
            vals_all[task] = []
        val_solvus_mse = list()
        val_tasks_mse = {name: [] for name in self.task_name}
        for epoch in range(epochs):
            task_commons = dict()
            for tn, task in enumerate(self.task_name):
                task_commons[task] = list()
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')

            if not self.multi_input:
                for batch_index in range(train_batch):
                    train_inputs, train_process, train_gts = self._process_data(train_loader)
                    train_preds = self.model([train_inputs,train_process])
                    train_preds = self.process_preds(train_preds)
                    train_losses = self._compute_loss(train_preds, train_gts)
                    self.meter.update(train_preds, train_gts)
                self.optimizer.zero_grad()
                w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()
            else:
                data_iterators = {task: iter(train_loader[task]) for task in self.task_name}
                for batch_index in range(train_batch):
                    train_losses = torch.zeros(self.task_num).to(self.device)
                    for tn, task in enumerate(self.task_name):
                        try:
                            train_batch_data_iter = data_iterators[task]
                            train_input, train_plus, train_gt = next(train_batch_data_iter)
                        except StopIteration:
                            # print("HERE!!!!")
                            # print(task)
                            data_iterators[task] = iter(train_loader[task])
                            train_batch_data_iter = data_iterators[task]
                            train_input, train_plus, train_gt = next(train_batch_data_iter)
                        # print("*******************")
                        # print(task)
                        # print(train_gt)
                        train_input = train_input.to(self.device, non_blocking=True)
                        train_plus = train_plus.to(self.device, non_blocking=True)
                        train_gt = train_gt.to(self.device, non_blocking=True)
                        # if task == "solvus":
                        #     print(train_gt)
                        #     print(batch_index)
                        # print(train_input.size())
                        # print(task)
                        ori_train_pred, commons = self.model([train_input, train_plus], task)
                        if params.weighting == 'EW' and params.lr == 0.001 and params.train_bs == 8 and params.T == (
                                1,) and params.weight_decay == 1e-4 and params.hiddens == 48:
                            task_commons[task].append((torch.squeeze(commons[task]),train_gt.tolist()))
                            # print(commons)
                        train_pred = ori_train_pred[task]
                        # print("Train_pred : %s" % train_pred)
                        # train_pred = self.process_preds(train_pred, task)
                        # print(tn)
                        # print(train_losses.size())
                        # print(self._compute_loss(ori_train_pred, train_gt, task))
                        # print(train_pred.size())
                        # print(train_gt.size())
                        train_losses[tn] = self._compute_loss(ori_train_pred, train_gt, task)[tn]
                        self.meter.update(train_pred, train_gt, task)
                    self.optimizer.zero_grad()
                    w = self.model.backward(train_losses, weights, **self.kwargs['weight_args'])
                    if w is not None:
                        self.batch_weight[:, epoch, batch_index] = w
                    self.optimizer.step()


                # for tn, task in enumerate(self.task_name):
                #     train_batch_data = train_loader[task]
                #     train_input, train_gt = self._process_data(train_batch_data)
                #     if task == "solvus":
                #         print(train_gt)
                #         print(batch_index)
                #     ori_train_pred,commons = self.model(train_input, task)
                #     if params.weighting == 'EW' and params.lr == 0.01 and params.train_bs == 64 and params.T == (1,) and params.weight_decay == 1e-4 and params.hiddens == 48:
                #
                #         task_commons[task].append(commons)
                #         # print(commons)
                #     train_pred = ori_train_pred[task]
                #     # print("Train_pred : %s" % train_pred)
                #     # train_pred = self.process_preds(train_pred, task)
                #     # print(tn)
                #     # print(train_losses.size())
                #     # print(self._compute_loss(ori_train_pred, train_gt, task))
                #     if task == "phase_class":
                #         train_losses[tn] = self._compute_loss(ori_train_pred, train_gt, task)[tn]
                #         self.meter.update(train_pred, train_gt, task)  # 得到self.abs_record, self.rel_record和self.bs
                #     else:
                #         train_losses[tn] = self._compute_loss(ori_train_pred, train_gt, task)[tn]
                #         self.meter.update(train_pred, train_gt, task)
                #     self.optimizer.zero_grad()
                #     w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                #     if w is not None:
                #         self.batch_weight[:, epoch, batch_index] = w
                #     self.optimizer.step()

            self.meter.record_time('end')
            self.meter.get_score()  # 获得self.abs_record, self.rel_record在每个batch中的加和平均值，得到self.result和self.loss_item
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            task_mse = self.meter.display(epoch=epoch, mode='train')  # 计算improvement并更新最好结果
            for tn, task in enumerate(self.task_name):
                train_all[task].append(task_mse[task])
            dataloaders_train_predict = dict()
            dataloaders_all_predict = dict()
            dataloaders_test_predict = dict()
            for task_id in range(len(data_train.all_dataset_task)):
                dataloaders_train_predict[data_train.task_names[task_id]] = torch.utils.data.DataLoader(
                    dataset=data_train.all_dataset_task[task_id],
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True)
                dataloaders_all_predict[data_train.task_names[task_id]] = torch.utils.data.DataLoader(
                    dataset=data_all.all_dataset_task[task_id],
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True)
                dataloaders_test_predict[data_test.task_names[task_id]] = torch.utils.data.DataLoader(
                    dataset=data_test.all_dataset_task[task_id],
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True)

            predict_train_mse, predict_results_train, predict_y_train, true_y_train, task_commons_train,test_inputs_all,a,b = self.test(
                dataloaders_train_predict)
            training_score = mtl_r2(predict_results_train, true_y_train)
            predict_test_mse, predict_results_test, predict_y_test, true_y_test, task_commons_test,test_inputs_all,a,b = self.test(
                dataloaders_test_predict)
            testing_score = mtl_r2(predict_results_test, true_y_test)
            pre_list_test = list()
            true_list_test = list()

            # for test in predict_results_test:
            #
            #     pre_list_test.append(test['density'].tolist()[0])
            # for test in true_y_test:
            #     true_list_test.append(test[0])
            # all_predicts.extend(pre_list_test)
            # all_trues.extend(true_list_test)

            # print("Predict:")
            # print(predict_results_test)
            # print("True:")
            # print(true_y_test)

            predict_all_mse, predict_results_all, predict_y_all, true_y_all, task_commons_all,test_inputs_all,a,b = self.test(dataloaders_all_predict)



            all_score = mtl_r2(predict_results_all, true_y_all)
            epoch2score.append((training_score, testing_score, all_score))
            self.meter.reinit()

            if dataloaders_test_predict is not None:
                self.meter.has_val = True
                val_mse,ori_predict_pred,predict_y,true_y,task_commons,test_inputs_all,all_predicts,all_trues = self.test(dataloaders_test_predict, epoch, mode='val',
                                            return_improvement=True)  # 比较self.result和self.base_result的区别
                for name in self.task_name:
                    vals_all[name].append(val_mse[name])
                # val_solvus_mse.append(val_improvement)
            # ave_improve = self.test(valid_dataloaders, epoch, mode='test')
            # if val_improvement :
            #     print("Improve is %f" % val_improvement)
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and dataloaders_test_predict is not None:
                    self.scheduler.step(val_mse)
                else:
                    self.scheduler.step()
            # if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
            #     torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
            #     print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
            # torch.save(self.model.state_dict(), r"C:\Users\Administrator\Desktop\wwr-files\multi-task\code\frame2_creep\files\density_phase\re_model\models-101.pt")
            #     print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best-244.pt')))
            if self.meter.best_result['epoch'] == 0:
                import pandas as pd
                test_set = data_test.all_dataset_task[1]
                comps_data = []
                plus_data = []
                y_data = []
                for di in range(len(test_set)):
                    comps_data.append(test_set[di][0].tolist())
                    plus_data.append(test_set[di][1].tolist())
                    y_data.append(test_set[di][2].tolist())
                comps_data = torch.tensor(comps_data,dtype=torch.float32,device=self.device)
                plus_data = torch.tensor(plus_data,dtype=torch.float32,device=self.device)
                y_data = torch.tensor(y_data,dtype=torch.float32,device=self.device)

                # explainer = shap.DeepExplainer(self.model, [[comps_data, plus_data], 'solvus'])
                # shap_values = explainer.shap_values([[comps_data, plus_data], 'solvus'])
                # shap.summary_plot(shap_values, [[comps_data, plus_data], 'solvus'], plot_type="bar", plot_size=[15, 10])

        self.meter.display_best_result()
        # if return_weight:
        #     return self.batch_weight

        return train_all, vals_all, task_commons, epoch2score,all_predicts,all_trues,true_y_all, task_commons_all

    def test(self, test_dataloaders, epoch=None, mode='test', return_improvement=False):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch.
        '''

        all_predicts = list()
        all_trues = list()

        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        # print(test_batch)
        self.model.eval()
        self.meter.record_time('begin')
        ori_predict_pred = list()
        true_y = list()
        predict_y = list()
        task_commons = dict()
        batch_numbs = 0
        test_inputs_all = list()
        with torch.no_grad():
            if not self.multi_input:
                test_losses = 0
                for batch_index in range(test_batch):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_inputs_all.append(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    self.meter.update(test_preds, test_gts)
                batches_ave_test = test_losses / batch_numbs
            else:
                data_iterators = {task: iter(test_loader[task]) for task in self.task_name}
                for tn, task in enumerate(self.task_name):
                    test_losses = 0
                    task_commons[task] = list()
                    for batch_index in range(test_batch[tn]):

                        try:
                            test_batch_data_iter = data_iterators[task]
                            test_input, test_plus, test_gt = next(test_batch_data_iter)
                        except StopIteration:
                            # print("HERE!!!!")
                            # print(task)
                            data_iterators[task] = iter(test_loader[task])
                            test_batch_data_iter = data_iterators[task]
                            test_input, test_plus, test_gt = next(test_batch_data_iter)
                        test_input = test_input.to(self.device, non_blocking=True)
                        test_plus = test_plus.to(self.device, non_blocking=True)
                        test_gt = test_gt.to(self.device, non_blocking=True)
                        # print(task)
                        ori_test_pred, commons = self.model([test_input,test_plus], task)
                        # for name in self.model.state_dict():
                        #     print(name)
                        # print("Model parameters:")
                        # print(self.model.state_dict()['commons.0.weight'])
                        pre_list_test = list()
                        true_list_test = list()
                        # print(ori_test_pred)
                        for test in test_gt:
                            true_list_test.append(test.tolist())
                        all_predicts.extend(pre_list_test)
                        all_trues.extend(true_list_test)
                        task_commons[task].append((torch.squeeze(commons[task]), test_gt.tolist()))
                        ori_predict_pred.append(ori_test_pred)

                        predict_y.append((test_input, ori_test_pred[task], test_gt))
                        true_y.append(test_gt.tolist())
                        test_pred = ori_test_pred[task]
                        test_pred = self.process_preds(test_pred)
                        test_loss = self._compute_loss(ori_test_pred, test_gt, task)[tn]
                        test_losses += test_loss
                        batch_numbs += 1
                        self.meter.update(test_pred, test_gt, task)


                    # batches_ave_test = test_losses/batch_numbs
                    # task_loss[task] = batches_ave_test
        self.meter.record_time('end')
        self.meter.get_score()
        task_mse = self.meter.display(epoch=epoch, mode=mode)
        # print("val_mse:",task_mse)
        improvement = self.meter.improvement
        self.meter.reinit()
        # if return_improvement:
        #     return improvement
        return task_mse,ori_predict_pred,predict_y,true_y,task_commons,test_inputs_all,all_predicts,all_trues

    def predict(self, predict_dataloaders):
        self.model.eval()
        self.meter.record_time('begin')
        predict_loader, predict_batch = self._prepare_dataloaders(predict_dataloaders)
        task_commons = dict()
        with torch.no_grad():
            if not self.multi_input:
                predict_inputs, predict_gts = self._process_data(predict_loader)
                ori_predict_pred,commons = self.model(predict_inputs)
            else:
                ori_predict_pred = list()
                for tn, task in enumerate(self.task_name):
                    if task in predict_loader.keys():
                        # print(task)
                        predict_inputs, predict_gts = self._process_data(predict_loader[task])
                        # print(self.model(predict_inputs, task))
                        out2, out =  self.model(predict_inputs, task)
                        task_commons[task] = out
                        ori_predict_pred.append(out2)
        return ori_predict_pred,task_commons

    def parse(self,predict_dataloaders):
        self.model.eval()
        self.meter.record_time('begin')
        predict_loader, predict_batch = self._prepare_dataloaders(predict_dataloaders)

        with torch.no_grad():
            if not self.multi_input:
                predict_inputs, predict_gts = self._process_data(predict_loader)
            else:
                for tn, task in enumerate(self.task_name):
                    if task in predict_loader.keys():
                        predict_inputs, predict_gts = self._process_data(predict_loader[task])

                        # 使用 SHAP 进行解释
                        explainer = shap.Explainer(self.model, predict_inputs.cpu().detach().numpy())
                        shap_values = explainer.shap_values(predict_inputs.cpu().detach().numpy())

                        # 可视化 SHAP 值
                        shap.summary_plot(shap_values, features=predict_inputs.cpu().detach().numpy(),
                                          feature_names=[f"Feature {i}" for i in range(task_features[task])], show=False)

                        # 添加标题和保存图像
                        plt.title('SHAP Values for Features')
                        plt.savefig(r'E:\desktop\shap\shap_values_plot_%s.png' % task)
                        plt.show()