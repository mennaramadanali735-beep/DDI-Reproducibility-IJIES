from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from utils import *
import pandas as pd
import csv
import random
from tqdm import tqdm
import copy
import numpy as np

def loadtrain(args):
    # train dataset
    df = pd.read_csv("dataset/Deng/drug_listxiao.csv")

    drug_list = df['drug_id'].tolist()
    zhongzi=args.zhongzi

    train =pd.read_csv('dataset/' +str('Deng/') + str(zhongzi) +'/ddi_training1xiao.csv')

    train_pos =[(h, t, r) for h, t, r in zip(train['d1'], train['d2'], train['type'])]  # 三元组 26056个
    # np.random.seed(args.seed)
    np.random.shuffle(train_pos)
    train_pos = np.array(train_pos)
    for i in range(train_pos.shape[0]):
        train_pos[i][0] = int(drug_list.index(train_pos[i][0]))
        train_pos[i][1] = int(drug_list.index(train_pos[i][1]))
        train_pos[i][2] = int(train_pos[i][2])
    label_list =[]
    for i in range(train_pos.shape[0]):
        label =np.zeros((65))
        label[int(train_pos[i][2])] = 1
        label_list.append(label)
    label_list =np.array(label_list)
    train_data= np.concatenate([train_pos, label_list],axis=1)

    # val dataset
    val = pd.read_csv('dataset/' + str('Deng/') + str(zhongzi) + '/ddi_validation1xiao.csv')

    val_pos = [(h, t, r) for h, t, r in zip(val['d1'], val['d2'], val['type'])]
    # np.random.seed(args.seed)
    np.random.shuffle(val_pos)
    val_pos = np.array(val_pos)
    for i in range(len(val_pos)):
        val_pos[i][0] = int(drug_list.index(val_pos[i][0]))
        val_pos[i][1] = int(drug_list.index(val_pos[i][1]))
        val_pos[i][2] = int(val_pos[i][2])
    label_list = []
    for i in range(val_pos.shape[0]):
        label = np.zeros((65))
        label[int(val_pos[i][2])] = 1
        label_list.append(label)
    label_list = np.array(label_list)
    val_data = np.concatenate([val_pos, label_list], axis=1)

    # test dataset
    test = pd.read_csv('dataset/' + str('Deng/') + str(zhongzi) + '/ddi_test1xiao.csv')
    test_pos = [(h, t, r) for h, t, r in zip(test['d1'] ,test['d2'], test['type'])]
    # np.random.seed(args.seed)
    np.random.shuffle(test_pos)
    test_pos= np.array(test_pos)
    print(test_pos[0])
    for i in range(len(test_pos)):
        test_pos[i][0] = int(drug_list.index(test_pos[i][0]))
        test_pos[i][1] = int(drug_list.index(test_pos[i][1]))
        test_pos[i][2] = int(test_pos[i][2])
    label_list = []
    for i in range(test_pos.shape[0]):
        label = np.zeros((65))
        label[int(test_pos[i][2])] = 1
        label_list.append(label)
    label_list = np.array(label_list)
    test_data = np.concatenate([test_pos, label_list], axis=1)
    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)


    return train_data ,val_data ,test_data