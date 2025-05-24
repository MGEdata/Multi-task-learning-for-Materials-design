import json
import optuna
from utils import set_device,set_random_seed,RegMetric,get_root_dir,AccuracyMetric
from loss import MSELoss,BCELoss
import os
import torch
from model import EncoderModel
import torch.nn as nn
from LibMTL import Trainer
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from LibMTL.config import LibMTL_args, prepare_args
import numpy as np
import logging
import sys
import pickle
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import random
import torch.nn.init as init

# OMP: Error
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


def pkl_load(path):
    # Load array from the file
    with open(path, 'rb') as file:
        loaded_array = pickle.load(file)
    return loaded_array

class CustomDataset(Dataset):
    def __init__(self, x, x_p, y):
        self.x = x
        self.x_p = x_p
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        if self.x_p.numel() != 0:
            x_plus = self.x_p[idx]
        else:
            x_plus = self.x_p
        y_sample = self.y[idx]
        return x_sample, x_plus, y_sample


def shuffle_generate_train_test(comp_path=r"./MTL/comps", prop_path =r"./MTL/props",ratio=[0.8,0.2],comb=(0,1)):
    comp_feat_numbs = 24
    feature_path = comp_path
    props_path = prop_path
    x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set = [], [], [], [], [], []
    x_train_process_set, x_valid_process_set, x_test_process_set = [],[],[]
    all_x, all_x_plus, all_y = [], [], []
    comps_data_path = os.listdir(feature_path)
    task_names = []
    for task_index in comb:
        comps_pkl_path = os.path.join(feature_path,comps_data_path[task_index])
        task_name = comps_data_path[task_index].replace("_comps.pkl", "")
        task_names.append(comps_data_path[task_index].replace("_comps.pkl", ""))
        props_pkl_path = os.path.join(props_path,task_name + "_prop.pkl")
        df_comp = pkl_load(comps_pkl_path)
        df_prop = pkl_load(props_pkl_path)
        num_rows = df_comp.shape[0]
        num_cols = df_comp.shape[1]
        ids = list(range(num_rows))
        random.seed(4)
        random.shuffle(ids)
        parts = [
            ids[:int(len(ids) * ratio[0])],
            ids[int(len(ids) * ratio[0]):]
        ]
        id_train, id_test = parts[0], parts[1]
        id_train, id_test = np.array(id_train), np.array(id_test)
        if num_cols > comp_feat_numbs+1:
            comp_train_df = df_comp[id_train,:comp_feat_numbs]
            process_train_df = df_comp[id_train,comp_feat_numbs:]
            prop_train_df = df_prop[id_train]
            comp_test_df = df_comp[id_test,:comp_feat_numbs]
            process_test_df = df_comp[id_test,comp_feat_numbs:]
            prop_test_df = df_prop[id_test]

            x_train_set.append(comp_train_df)
            y_train_set.append(prop_train_df)
            x_test_set.append(comp_test_df)
            y_test_set.append(prop_test_df)
            all_x.append(df_comp[:,:comp_feat_numbs])
            all_x_plus.append(df_comp[:,comp_feat_numbs:])
            all_y.append(df_prop)
            x_train_process_set.append(process_train_df)
            x_test_process_set.append(process_test_df)
        else:
            comp_train_df = df_comp[id_train]
            prop_train_df = df_prop[id_train]
            comp_test_df = df_comp[id_test]
            prop_test_df = df_prop[id_test]
            x_train_set.append(comp_train_df)
            y_train_set.append(prop_train_df)
            x_test_set.append(comp_test_df)
            y_test_set.append(prop_test_df)
            all_x.append(df_comp)
            all_x_plus.append([])
            all_y.append(df_prop)
            x_train_process_set.append([])
            x_test_process_set.append([])

    return all_x, all_x_plus, all_y, x_train_set, x_test_set, y_train_set,  y_test_set, task_names, x_train_process_set, x_test_process_set    # id_train, id_valid, id_test,


def get_true():
    import xlrd
    # 读取.xlsx文件
    file_path = r"E:\Text_mining\Work3\code\multi_task_embeds\multi_task\no_embed_six\code\true_y\task_trues.xlsx"
    workbook = xlrd.open_workbook(file_path)
    # 选择第一个工作表
    sheet = workbook.sheet_by_index(0)

    # 读取第一行数据作为键
    keys = [sheet.cell_value(0, col) for col in range(sheet.ncols)]

    # 初始化字典
    data_dict = {key: [] for key in keys}
    # 从第二行开始，读取每列数据
    for col in range(sheet.ncols):
        for row in range(1, sheet.nrows):
            if sheet.cell_value(row, col)!="":
                data_dict[keys[col]].append(sheet.cell_value(row, col))
    return data_dict


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def scores(name='mse', predicts=list(), trues=list()):
    if name == 'mse':
        out_score = mean_squared_error(predicts, trues)
    elif name == 'rmse':
        out_score = np.sqrt(mean_squared_error(predicts, trues))
    elif name == 'r2':
        out_score = r2_score(trues, predicts)

    return out_score


def get_predictions(comb=(0,1)):
    # task_data_path = os.listdir(r"E:\Text_mining\Work3\code\multi_task_embeds\multi_task\data\prediction")
    comps_numb = 24
    task_data_path = os.listdir(r"E:\Text_mining\Work3\code\multi_task_embeds\multi_task\no_embed_six\no_embed_two\files\prediction")
    task_names = []
    features = []
    all_numbsers = []
    for task_index in comb:
        pkl_path = os.path.join(r"E:\Text_mining\Work3\code\multi_task_embeds\multi_task\no_embed_six\no_embed_two\files\prediction", task_data_path[task_index])
        comp_array = pkl_load(pkl_path)
        data_numbs = comp_array.shape[0]
        all_numbsers.append(data_numbs)
        features.append(comp_array)
        task_names.append(task_data_path[task_index].replace("_comps.pkl", ""))
    return features,task_names,all_numbsers



class GetData(Dataset):
    """

    """
    def __init__(self,mode='train',comb=(1,2),comp_p=r"./MTL/comps",prop_p=r"./MTL/props"):
        self.mode = mode
        all_x, all_x_plus, all_y, x_train_set, x_test_set, y_train_set,  y_test_set, task_names, x_train_process_set, x_test_process_set = shuffle_generate_train_test(comp_path=comp_p, prop_path =prop_p, comb=comb) # id_train, id_valid, id_test,
        self.task_names = task_names
        new_data, task_names, all_numbsers = get_predictions(comb=comb)
        # read the data file
        self.all_dataset_task = list()
        if self.mode == 'train':
            for task_id in range(len(x_train_set)):
                x = x_train_set[task_id]
                x_plus = x_train_process_set[task_id]
                y = y_train_set[task_id]
                x = torch.tensor(np.array(x))
                x_plus = torch.tensor(np.array(x_plus))
                y = torch.tensor(np.array(y))
                dataset_task = CustomDataset(x, x_plus, y)
                self.all_dataset_task.append(dataset_task)

        elif self.mode == 'test':
            for task_id in range(len(x_train_set)):
                x = x_test_set[task_id]
                x_plus = x_test_process_set[task_id]
                y = y_test_set[task_id]
                x = torch.tensor(np.array(x))
                x_plus = torch.tensor(np.array(x_plus))
                y = torch.tensor(np.array(y))
                dataset_task = CustomDataset(x, x_plus, y)
                self.all_dataset_task.append(dataset_task)
        elif self.mode == "all":
            for task_id in range(len(x_train_set)):
                x = all_x[task_id]
                x_plus = all_x_plus[task_id]
                y = all_y[task_id]
                x = torch.tensor(np.array(x))
                x_plus = torch.tensor(np.array(x_plus))
                y = torch.tensor(np.array(y))
                dataset_task = CustomDataset(x, x_plus, y)
                self.all_dataset_task.append(dataset_task)
        elif self.mode == "new":
            for task_id in range(len(x_train_set)):
                x = torch.tensor(np.array(new_data[task_id])[:24])
                x_plus = torch.tensor(np.array(new_data[task_id])[24:])
                y = all_numbsers[task_id]
                y = torch.tensor(np.array(list(range(y))))
                dataset_task = CustomDataset(x, x_plus, y)
                self.all_dataset_task.append(dataset_task)


class Params:
    def __init__(self,
                 features=24,
                 seed=2,
                 gpu_id='0',
                 train_mode='train',
                 lr=1e-2,  # learning rate for all types of optim
                 nbs=64,
                 train_bs= 8,
                 test_bs= 8,
                 epochs= 100,
                 save_path=os.path.join(r".\results",r"model"),
                 load_path=os.path.join(r".\multi_task","existed_model"),
                 weighting='EW', # [EW, UW, GradNorm, GLS, RLW, MGDA, PCGrad, GradVac, CAGrad, GradDrop, DWA, IMTL']
                 arch='HPS',  # ['HPS', 'MTAN']
                 rep_grad=False,  # help='computing gradient for representation or sharing parameters'
                 multi_input=True,  # 'whether each task has its own input data'
                 optim='adam',  # optimizer for training, option: adam, sgd, adagrad, rmsprop
                 momentum = 0.8,  # help='momentum for sgd'
                 weight_decay = 1e-3,  # weight decay for all types of optim
                 scheduler = 'step',
                 step_size = 100,  # 'step size for StepLR'
                 gamma = 0.5,  # gamma for StepLR
                 T=1.0,
                 mgda_gn = 'none',
                 GradVac_beta = 0.5,  # 'beta for GradVac')
                 GradVac_group_type = 0,  # 'parameter granularity for GradVac (0: whole_model; 1: all_layer; 2: all_matrix)')
                 alpha = 1.5,  # 'alpha for GradNorm')
                 leak = 0.0,  # 'leak for GradDrop')
                 calpha = 0.5,  # 'calpha for CAGrad')
                 rescale = 1,  # 'rescale for CAGrad')
                 update_weights_ever = 1,  # update_weights_every for Nash_MTL
                 optim_niter = 20,  # optim_niter for Nash_MTL'
                 max_norm = 1.0,  # 'max_norm for Nash_MTL'
                 MoCo_beta = 0.5,  # MoCo_beta for MoCo'
                 MoCo_beta_sigma = 0.5,  # MoCo_beta_sigma for MoCo')
                 MoCo_gamma = 0.1,  # 'gamma for MoCo'
                 MoCo_gamma_sigma = 0.5,  # 'MoCo_gamma_sigma for MoCo'
                 MoCo_rho = 0,  # MoCo_rho for MoCo'
                 img_size = [],  # help='image size for CGC'
                 num_experts = [],  # the number of experts for sharing and task-specific, number = len(task) +1 ,第一位为所有任务共享的expert
                 num_nonzeros = 2,  # 'num_nonzeros for DSelect-k')
                 kgamma = 1.0,  # gamma for DSelect-k'
                 hiddens = 6
                 ):
        self.features = features
        self.seed = seed
        self.gpu_id = gpu_id
        self.weighting = weighting
        self.nbs=nbs
        self.arch = arch
        self.rep_grad = rep_grad
        self.optim = optim
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.multi_input = multi_input
        self.train_mode = train_mode
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.epochs = epochs
        self.save_path = save_path
        self.load_path = load_path
        self.T = T,
        self.mgda_gn = mgda_gn,
        self.GradVac_beta = GradVac_beta,  # 'beta for GradVac')
        self.GradVac_group_type = GradVac_group_type,  # 'parameter granularity for GradVac (0: whole_model; 1: all_layer; 2: all_matrix)')
        self.alpha = alpha,  # 'alpha for GradNorm')
        self.leak = leak,  # 'leak for GradDrop')
        self.calpha = calpha,  # 'calpha for CAGrad')
        self.rescale = rescale,  # 'rescale for CAGrad')
        self.update_weights_ever = update_weights_ever,  # update_weights_every for Nash_MTL
        self.optim_niter = optim_niter,  # optim_niter for Nash_MTL'
        self.max_norm = max_norm,  # 'max_norm for Nash_MTL'
        self.MoCo_beta = MoCo_beta,  # MoCo_beta for MoCo'
        self.MoCo_beta_sigma = MoCo_beta_sigma,  # MoCo_beta_sigma for MoCo')
        self.MoCo_gamma = MoCo_gamma,  # 'gamma for MoCo'
        self.MoCo_gamma_sigma = MoCo_gamma_sigma,  # 'MoCo_gamma_sigma for MoCo'
        self.MoCo_rho = MoCo_rho,  # MoCo_rho for MoCo'
        self.img_size = img_size,  # help='image size for CGC'
        self.num_experts = num_experts,  # the number of experts for sharing and task-specific, number = len(task) +1 ,第一位为所有任务共享的expert
        self.num_nonzeros = num_nonzeros,  # 'num_nonzeros for DSelect-k')
        self.kgamma = kgamma,  # gamma for DSelect-k'
        self.hiddens = hiddens


def main(params,data_train,data_valid,data_test,data_all,num_layers,data_new):
    fig_id=1
    kwargs, optim_param, scheduler_param = prepare_args(params)
    # define tasks , 决定task的多少
    task_dict = {
                 'solvus': {'metrics':['abs_err', 'rel_err'],
                             'metrics_fn': RegMetric(),
                             'loss_fn': MSELoss(),
                             'weight': [0, 0]
                             },
                'solidus': {'metrics': ['abs_err', 'rel_err'],
                   'metrics_fn': RegMetric(),
                   'loss_fn': MSELoss(),
                   'weight': [0, 0]
                   },
                'liquidus': {'metrics': ['abs_err', 'rel_err'],
                           'metrics_fn': RegMetric(),
                           'loss_fn': MSELoss(),
                           'weight': [0, 0]
                           },
                'density': {'metrics': ['abs_err', 'rel_err'],
                           'metrics_fn': RegMetric(),
                           'loss_fn': MSELoss(),
                           'weight': [0, 0]
                           },
                'size': {'metrics': ['abs_err', 'rel_err'],
                           'metrics_fn': RegMetric(),
                           'loss_fn': MSELoss(),
                           'weight': [0, 0]
                           },
                'phase_class': {'metrics': ['accuracy'],
                   'metrics_fn': AccuracyMetric(),
                   'loss_fn': BCELoss(),
                   'weight': [2]
                   },
                 }

    set_random_seed(seed=0)
    encoders = nn.Sequential(
        nn.Linear(24, params.hiddens),
        nn.ReLU(),
        nn.Linear(params.hiddens, params.hiddens),
        nn.ReLU(),
        nn.Linear(params.hiddens, int(params.hiddens / 2)),
        nn.ReLU(),
        nn.Linear(int(params.hiddens / 2), int(params.hiddens / 2)),
        nn.ReLU(),
    )
    for module_1 in encoders:
        if isinstance(module_1, nn.Linear):
            init.xavier_normal_(module_1.weight)
    set_random_seed(seed=0)

    attention = nn.ModuleDict(
        {
            'solvus': None,
            'solidus': None,
            'liquidus': None,
            'density': None,
            'size': None,
            'phase_class': None,
            'creep_cluster0': nn.MultiheadAttention(params.hiddens, 4),  # nn.MultiheadAttention(params.hiddens, 16),
            'creep_cluster1': nn.MultiheadAttention(params.hiddens, 8),
        }
    )
    num_out_channels = {'density': 1, 'liquidus': 1, 'solidus': 1, 'solvus': 1, 'size': 1,'phase_class': 1}

    class trainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, lr,encoder_class,
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, weight_decay,**kwargs):
            super(trainer, self).__init__(task_dict=task_dict,
                                             weighting=weighting_method.__dict__[weighting],
                                             architecture=architecture_method.__dict__[architecture],
                                             lr=lr, # learning rate for all types of optim
                                             encoder_class=encoder_class,
                                             decoders=decoders,
                                             rep_grad=rep_grad,
                                             multi_input=multi_input,
                                             optim_param=optim_param,
                                             scheduler_param=scheduler_param,
                                             weight_decay=weight_decay,
                                             **kwargs)
    set_random_seed(seed=0)

    in_features = int(params.hiddens/2)  # 初始的输入特征数
    layers = nn.Sequential()
    for _ in range(num_layers):
        layers.append(nn.Linear(in_features, in_features // 2))
        layers.append(nn.ReLU())
        in_features //= 2

    # 输出层
    layers.append(nn.Linear(in_features, num_out_channels['phase_class']))
    layers.append(nn.Sigmoid())



    decoders = nn.ModuleDict(
        {
            'solvus': nn.Sequential(
                nn.Linear(int(params.hiddens/2), int(params.hiddens/4)),
                nn.ReLU(),
                nn.Linear(int(params.hiddens/4), num_out_channels['solvus'])
            ),
            'solidus': nn.Sequential(
                nn.Linear(int(params.hiddens / 2), int(params.hiddens / 4)),
                nn.ReLU(),
                nn.Linear(int(params.hiddens / 4), num_out_channels['solidus'])
            ),
            'liquidus': nn.Sequential(
                nn.Linear(int(params.hiddens / 2), int(params.hiddens / 4)),
                nn.ReLU(),
                nn.Linear(int(params.hiddens / 4), num_out_channels['liquidus'])
            ),
            'size': nn.Sequential(
                nn.Linear(int(params.hiddens / 2)+4, int(params.hiddens / 4)),
                nn.ReLU(),
                nn.Linear(int(params.hiddens / 4), num_out_channels['size'])
            ),
            'density': nn.Sequential(
                nn.Linear(int(params.hiddens / 2), int(params.hiddens / 4)),
                nn.ReLU(),
                nn.Linear(int(params.hiddens / 4), num_out_channels['density'])
            ),
            'phase_class': layers

        }
    )

    model = trainer(task_dict=task_dict,
                    weighting=params.weighting,
                    architecture=params.arch,
                    lr=params.lr,
                    encoder_class=encoders,
                    decoders=decoders,
                    attention=attention,
                    rep_grad=params.rep_grad,
                    multi_input=params.multi_input,
                    optim_param=optim_param,
                    scheduler_param=scheduler_param,
                    save_path=params.save_path,
                    load_path=params.load_path,
                    weight_decay=params.weight_decay,
                    **kwargs
                    )
    dataloaders_train = dict()
    dataloaders_test = dict()
    dataloaders_all = dict()
    dataloaders_new = dict()
    for task_id in range(len(data_train.all_dataset_task)):
        dataloaders_train[data_train.task_names[task_id]] = torch.utils.data.DataLoader(
            dataset=data_train.all_dataset_task[task_id],
            batch_size=params.train_bs,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True)
        dataloaders_test[data_test.task_names[task_id]] = torch.utils.data.DataLoader(
            dataset=data_test.all_dataset_task[task_id],
            batch_size=400,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False)
        dataloaders_all[data_all.task_names[task_id]] = torch.utils.data.DataLoader(
            dataset=data_all.all_dataset_task[task_id],
            batch_size=400,
            shuffle=False,
            num_workers=0,
            pin_memory=True)
        dataloaders_new[data_new.task_names[task_id]] = torch.utils.data.DataLoader(
            dataset=data_new.all_dataset_task[task_id],
            batch_size=400,
            shuffle=False,
            num_workers=0,
            pin_memory=True)
    train_mse,val_mse,task_commons_train,epoch2score,all_predicts,all_trues = model.train(dataloaders_train,data_train,data_test,data_all, val_dataloaders=dataloaders_test, epochs=params.epochs,fig_id=fig_id,params=params)
    mse, valid_results, predict_y, true_y, task_commons_valid,test_inputs_all,a,b = model.test(dataloaders_test)
    valid_y = dict()
    for name, info in dataloaders_test.items():
        valid_y[name] = list()
        for x, x_p, y in info:
            valid_y[name].extend(y.tolist())

    train_y = dict()
    for name, info in dataloaders_train.items():
        train_y[name] = list()
        for x, x_p, y in info:
            train_y[name].extend(y.tolist())

    all_y = dict()
    for name, info in dataloaders_all.items():
        all_y[name] = list()
        for x, x_p, y in info:
            all_y[name].extend(y.tolist())

    mse, predict_results_all, true_x, true_y, task_commons_all,test_inputs_all,a,b = model.test(dataloaders_all)
    task_predict_score = dict()
    loss_score = 0
    for task in valid_results:
        for name, predict in task.items():
            pred_data = torch.squeeze(predict).tolist()
            true_data = valid_y[name]
            mse = scores(name='mse', predicts=pred_data, trues=true_data)
            r2 = scores(name='r2', predicts=pred_data, trues=true_data)
            relas = list()
            if name != "phase_class":
                for data_i in range(len(pred_data)):
                    rela = abs(pred_data[data_i] - true_data[data_i]) / true_data[data_i]
                    relas.append(rela)
                task_predict_score[name] = (mse, r2, np.mean(relas))
                loss_score += mse
            else:
                predicted_labels = (torch.squeeze(predict) >= 0.5).int()
                predicted_labels = torch.flatten(predicted_labels)
                count = 0
                # 使用循环比较并计数相同的元素
                for list_element, tensor_element in zip(true_data, predicted_labels):
                    if list_element == tensor_element.item():
                        count += 1
                accuracy = count / len(pred_data)
                task_predict_score[name] = (accuracy, None, None)


    return train_mse,val_mse,task_predict_score,task_commons_train,predict_results_all,epoch2score,all_predicts,all_trues,predict_y #,np.mean(relas),solvus_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set device
set_device(gpu_id="0")


colors_ = ["#86A789","#4D4C7D","#F99417","#DA0C81","#0802A3","#FF4B91","#FF7676","#FFCD4B","#D6D46D","#F4DFB6","#DE8F5F","#9A4444","#F8BDEB","#BEADFA","#445D48"]
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from tqdm import tqdm

# 获取 CSS4 规范中定义的颜色集合
colors_ = list(mcolors.CSS4_COLORS.values())
set_random_seed(seed=0)


epoch = 100
fig_id = 1

params_ = list()
for weight in ["EW", "UW"]:
    for lr in [1e-3,1e-2]:#,1e-3,1e-2
        for train_bs in [4,8,16]:#,24,32
            for weight_decay in [1e-3,1e-2]:#,1e-3,1e-2
                for hidden in [32,56,72]: # ,24,48
                    for num_layers in [0,1,2]:
                        params_.append((weight,lr,train_bs,weight_decay,hidden,num_layers))


from itertools import permutations
from itertools import combinations


# 生成两两元素的排列组合
comb = [0, 1, 2, 3, 4, 5]
all_combs = list()
# 生成不同数量元素的所有组合
for r in range(1, len(comb) + 1):
    all_combinations = list(combinations(comb, r))
    for x in all_combinations:
        all_combs.append(list(x))

all_combs.append(comb)

comp_p=r"./MTL/comps"
prop_p=r"./MTL/props"

# 输出结果
comb_results = dict()
for comb in tqdm(all_combs):
    data_train = GetData(mode='train',comb=comb,comp_p=comp_p,prop_p=prop_p)
    data_valid = GetData(mode='valid',comb=comb,comp_p=comp_p,prop_p=prop_p)
    data_test = GetData(mode='test',comb=comb,comp_p=comp_p,prop_p=prop_p)
    data_all = GetData(mode='all',comb=comb,comp_p=comp_p,prop_p=prop_p)
    data_new = GetData(mode='new',comb=comb,comp_p=comp_p,prop_p=prop_p)
    result = dict()
    result[str(data_new.task_names)] = list()
    param2all = dict()
    para2r2 = dict()
    params_id = 0
    for param in tqdm(params_):
        param2loss_solvus = dict()
        param2loss_solidus = dict()
        param2loss_liquidus = dict()
        param2loss_density = dict()
        param2loss_size = dict()
        param2phase = dict()
        param2mre = dict()
        param2fig = dict()
        id2score = dict()
        id2train = dict()
        id2val = dict()
        id2all_score = dict()
        creep_val_mse = list()
        fig2params = dict()
        # layer = param[0]
        layer = 0
        weight = param[0]
        lr = param[1]
        train_bs = param[2]
        weight_decay = param[3]
        hidden = param[4]
        num_layers = param[5]
        predicts = list()
        train_m = list()
        val_m = list()
        predict_m = list()

        params = Params(train_mode='train',
                        load_path=None,
                        epochs=epoch,
                        weighting=weight,
                        momentum=0.8,
                        lr=lr,
                        train_bs=train_bs,
                        hiddens=hidden,
                        T=1,
                        weight_decay=weight_decay,
                        )
        param2all[str((weight, lr, train_bs, weight_decay, hidden))] = 0
        try:
            train_mse,val_mse,task_predict_score,task_commons_train,predict_results_all,epoch2score,all_predicts,all_trues,predict_y = main(params,data_train,data_valid,data_test,data_all,num_layers,data_new)
        except ValueError as e:
            continue
        creep_loss = []
        size_loss = []
        solvus_loss = []
        density_loss = []
        liquidus_loss = []
        solidus_loss = []
        size_loss = []
        phase_r2 = []
        creep_r2 = []
        size_r2 = []
        name_test_loss = dict()
        for data in epoch2score:
            for name in data[1].keys():
                if name in name_test_loss.keys():
                    name_test_loss[name].append(data[1][name][1])# 最后一个为0表示精度和，为1表示误差和
                else:
                    name_test_loss[name] = list()
                    name_test_loss[name].append(data[1][name][1])
        train_loss = []
        test_loss = []
        all_loss = []
        loss = list()
        for name in name_test_loss.keys():
            param2all[str((weight, lr, train_bs, weight_decay, hidden,))] += name_test_loss[name][-1]
        name_test_r2 = dict()
        r2_sum = list()
        for data in epoch2score:
            r2 = 0
            for name in data[0].keys():
                if name != "phase_class":
                    if name in name_test_r2.keys():
                        name_test_r2[name].append(data[1][name][0])
                    else:
                        name_test_r2[name] = list()
                        name_test_r2[name].append(data[1][name][0])
                    r2 += data[1][name][0]
            r2_sum.append(r2)
        para2r2[str((weight, lr, train_bs, weight_decay, hidden))] = max(r2_sum)

        fig, ax1 = plt.subplots(dpi=300)
        ax2 = ax1.twinx()

        plot_train_mse = dict()
        for task in data_train.task_names:
            plot_train_mse[task] = list()
        for task in data_train.task_names:
            for data in train_mse[task]:
                plot_train_mse[task].append(data)

        i = 0
        for task in data_train.task_names:
            if task != "phase_class":
                ax1.plot(list(range(epoch)), plot_train_mse[task], 'b-', linewidth=2.0, label='train-%s' % task,
                         color='C' + str(i))
            else:
                ax2.plot(list(range(epoch)), plot_train_mse[task], 'r-', linewidth=2.0, label='train-%s' % task,
                         color='C' + str(i))
            i += 1

        for task in data_train.task_names:
            y = val_mse[task]
            ry = list()

            if task != "phase_class":
                for ys in y:
                    ry.append(np.array(ys))
                ax1.plot(list(range(epoch)), ry, 'b-', linewidth=2.0, label='val-%s' % task, color='C' + str(i))
            else:
                for ys in y:
                    ry.append(np.array(ys))
                ax2.plot(list(range(epoch)), ry, 'r-', linewidth=2.0, label='val-%s' % task, color='C' + str(i))
            i += 1

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE')
        ax2.set_ylabel('Accuracy')
        ax1.set_ylim((0, 10000))
        ax2.set_ylim((0, 1.2))
        path_save = r".\%s.jpg" % params_id
        plt.savefig(path_save)
        params_id += 1

    comb_results[str(comb)] = (comb,param2all)

