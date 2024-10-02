import pickle
import csv
import torch
import numpy as np
import pickle
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import optuna

def convert_to_wt_percent(input_csv, output_csv, atomic_mass_dict):
    # 读取输入CSV文件
    all_info = pd.read_csv(input_csv)

    rows = all_info.values.tolist()
    elements = list(atomic_mass_dict.keys())
    # 提取元素含量和计算wt.%的值
    for row in rows:
        at_sum = 0
        for i in range(24):
            element_content_at_percent = float(row[i])
            element = elements[i]
            atomic_mass = atomic_mass_dict[element]
            at_sum += element_content_at_percent * atomic_mass
        for i in range(24):  # 假设前24列是合金数据
            element_content_at_percent = float(row[i])
            element = elements[i]
            atomic_mass = atomic_mass_dict[element]
            wt_percent = (element_content_at_percent * atomic_mass) * 100 / at_sum
            row[i] = wt_percent

    # 将转换后的数据写入新的CSV文件
    with open(output_csv, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(rows)


# 字典包含元素的原子质量（示例数据，需要根据实际情况更新）
atomic_mass_dict = {
    'Co': 58.933195,  # C
    'Al': 26.981539,   # H
    'W': 183.84,   # H
    'Ni': 58.6934,   # H
    'Ti': 47.867,   # H
    'Cr': 51.9961,   # H
    'Ge': 72.630,   # H
    'Ta': 180.947,   # H
    'B': 10.806,   # H
    'Mo': 95.95,   # H
    'Re': 186.207,   # H
    'Nb': 92.906,   # H
    'Mn': 54.938,   # H
    'Si': 28.084,   # H
    'V': 50.9415,   # H
    'Fe': 55.845,   # H
    'Zr': 91.224,   # H
    'Hf': 178.486,   # H
    'Ru': 101.07,   # H
    'Ir': 192.22,   # H
    'La': 138.905,   # H
    'Y': 88.905,   # H
    'Mg': 	24.305,   # H
    'C': 12.011,   # H

    # 其他元素的原子质量
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed):
    r"""Set the random seed for reproducibility.

    Args:
        seed (int, default=0): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True # 将 CuDNN 的随机性设置为确定性模式，即启用确定性算法。
        torch.backends.cudnn.benchmark = False

def pkl_process(path,act,data=None):
    if act == 'load':
        with open(path, 'rb') as f:
            info = pickle.load(f)
        return info
    elif act == 'save':
        with open(path, 'wb') as file:
            pickle.dump(data, file)

import json

def json_save_load(mode='save',obj={},json_path=r"a"):
    if mode=="save":
        with open(json_path,"w", encoding='utf-8') as f:
            f.write(json.dumps(obj, indent=2))
    elif mode=="load":
        with open(json_path,'r',encoding='utf8')as fp:
            json_data = json.load(fp)
        return json_data

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
        nn.Linear(24, 72),
        nn.ReLU(),
        nn.Linear(72, 72),
        nn.ReLU(),
        nn.Linear(72, 36),
        nn.ReLU(),
        nn.Linear(36, 36),
    )
        self.decoders = nn.ModuleDict(
            {
                'solvus': nn.Sequential(
                    nn.Linear(36, 18),
                    nn.ReLU(),
                    nn.Linear(18,1)
                ),
                'solidus': nn.Sequential(
                    nn.Linear(36, 18),
                    nn.ReLU(),
                    nn.Linear(18, 1)
                ),
                'liquidus': nn.Sequential(
                    nn.Linear(36, 18),
                    nn.ReLU(),
                    nn.Linear(18, 1)
                ),
                'size': nn.Sequential(
                    nn.Linear(36 + 4, 18),
                    nn.ReLU(),

                    nn.Linear(18, 1)
                ),
                'density': nn.Sequential(
                    nn.Linear(36, 18),
                    nn.ReLU(),
                    # nn.Linear(18, 18),
                    # nn.ReLU(),
                    nn.Linear(18, 1)
                ),
                'oxidation_gain': nn.Sequential(
                    nn.Linear(36 + 4, 18),
                    nn.ReLU(),

                    nn.Linear(18, 1)
                ),
                'creep': nn.Sequential(
                    nn.Linear(36 + 13, 36),
                    nn.ReLU(),
                    nn.Linear(36, 18),
                    nn.ReLU(),
                    nn.Linear(18, 18),
                    nn.ReLU(),
                    nn.Linear(18, 1)
                ),
                'phase_class': nn.Sequential(
                    # nn.Linear(int(params.hiddens/2), int(params.hiddens/2)),
                    # nn.ReLU(),
                    # nn.Linear(36, 36),
                    # nn.ReLU(),
                    # nn.Linear(36, 18),
                    # nn.ReLU(),
                    nn.Linear(36, 1),
                    nn.Sigmoid()
                ),
                # 'oxidation_gain': nn.Sequential(
                #     nn.Linear(36 + 2, 18),
                #     nn.ReLU(),
                #
                #     nn.Linear(18, 1)
                # ),
            }
        )

    def forward(self, x, task):
        hidden = self.encoder(x)
        print(hidden)
        x = self.decoders[task](hidden)
        return x,hidden

model = Model()
model_path = r'\models-115.pt'
model.load_state_dict(torch.load(model_path))
model.to(device=device)
task = "solidus"
input_csv_file = r'predict.csv'  # 你的输入CSV文件名
output_csv_file = r'predict_wt.csv'  # 新的输出CSV文件名
# 执行转换
convert_to_wt_percent(input_csv_file, output_csv_file, atomic_mass_dict)
to_predicts = pd.read_csv(output_csv_file,header=None)
alloy_comps = to_predicts.values
plus = alloy_comps[:,24:]
alloy_comps = torch.tensor(alloy_comps,dtype=torch.float32,device=device)
plus = torch.tensor(plus,dtype=torch.float32,device=device)
ori_train_pred,hidden = model(alloy_comps, task)

embeds_all = pkl_process(act='load',path=r"embeds_full_115.pkl")

pca = PCA(n_components=2)

X_embeddeds = pca.fit_transform(embeds_all)
pos = pca.transform(hidden.clone().detach().cpu().numpy())
