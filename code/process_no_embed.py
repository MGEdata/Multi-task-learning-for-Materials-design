import json
import os
import pandas as pd
import pickle
import numpy as np
import random

def json_load(path):
    with open(path, 'r') as json_file:
        loaded_data = json.load(json_file)
    return loaded_data

def json_save_commom(obj,path):
    with open(path,"w", encoding='utf-8')  as f:
        f.write(json.dumps(obj, indent=2))

def pkl_load(path):
    # Load array from the file
    with open(path, 'rb') as file:
        loaded_array = pickle.load(file)
    return loaded_array


data_path=r"C:\Users\Administrator\Desktop\wwr-files\multi-task\code\frame2_creep\files\solidus_liquidus\add_csv"
output_path=r"C:\Users\Administrator\Desktop\wwr-files\multi-task\code\frame2_creep\files\solidus_liquidus\comps_add"
csv_paths = os.listdir(data_path)
data_dicts = list()
for csv_p in csv_paths:
    name = csv_p.replace(".csv","")
    csv_full_path = os.path.join(data_path,csv_p)
    prop_array = pd.read_csv(csv_full_path)
    cols_name = prop_array.columns.tolist()
    cols_name_elements = cols_name[:-1]
    rows = prop_array.shape[0]
    prop_data_dict = {}
    prop_data_dict[name] = list()
    comps_data = list()
    re_prop_data = list()
    for data_i in list(range(rows)):
        single_data = prop_array[data_i:data_i+1]
        prop_numb = single_data.iloc[0,-1]
        comps = single_data.iloc[0,:-1]
        comps = comps.values
        comps_data.append(comps)
        re_data = list()
        re_prop_data.append(prop_numb)

    prop_data_dict[name].append(comps_data)
    prop_data_dict[name].append(re_prop_data)
    comps_data,re_prop_data = np.array(comps_data),np.array(re_prop_data)
    with open(os.path.join(output_path, r'%s_comps.pkl' % name), 'wb') as file:
        pickle.dump(comps_data, file)
    with open(os.path.join(output_path, r'%s_prop.pkl' % name), 'wb') as file:
        pickle.dump(re_prop_data, file)

# for predict
# data_path=r"E:\Text_mining\Work3\code\multi_task_embeds\multi_task\no_embed_six\frame2_creep\files\prediction_csv"
# output_path=r"E:\Text_mining\Work3\code\multi_task_embeds\multi_task\no_embed_six\frame2_creep\files\prediction"
# csv_paths = os.listdir(data_path)
# data_dicts = list()
# task2data = dict()
# for csv_p in csv_paths:
#     name = csv_p.replace(".csv","")
#     csv_full_path = os.path.join(data_path,csv_p)
#     prop_array = pd.read_csv(csv_full_path)
#     cols_name = prop_array.columns.tolist()
#     cols_name_elements = cols_name
#     rows = prop_array.shape[0]
#     comps_data = list()
#     prop_data_dict = {}
#     prop_data_dict[name] = list()
#
#     re_prop_data = list()
#     for data_i in list(range(rows)):
#         single_data = prop_array[data_i:data_i + 1]
#         comps = single_data.values.squeeze()
#         comps_data.append(comps)
#         re_data = list()
#     task2data[name] = comps_data
#     comps_data = np.array(comps_data)
#     with open(os.path.join(output_path, r'%s_comps.pkl' % name), 'wb') as file:
#         pickle.dump(comps_data, file)




