import copy
import time
import torch
import numpy as np
import torch.nn as nn
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve,auc
import os
import random
from tqdm import tqdm
import torch.nn.functional as F


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)

def train_model(model,device, optimizer, edge_index,label_list, train_loader_nol, train_loader, val_loader, test_loader, args):
    m = torch.nn.Sigmoid() 
    loss_fct=torch.nn.CrossEntropyLoss() 
    
    max_auc1 = 0

    max_t1 = 0
    patience = 25  


    # Train model
    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')
    stoping = 0
    
    # with torch.set_grad_enabled(True):
    model.cl_model.eval()  
    with torch.no_grad():
        pbar = tqdm(train_loader_nol, desc="Iteration", disable=True)
        for step, batch in enumerate(pbar):
            batch = batch.to(device)
            emb_2d_low, emb_2d_high, emb_1d_low, emb_1d_high = model.cl_model(batch)
            # low and high
            out_2d = (emb_2d_low + emb_2d_high) / 2
            out_1d = (emb_1d_low + emb_1d_high) / 2
            all_drug_feat = torch.concat((out_1d, out_2d), dim=1)
        all_drug_feat = all_drug_feat.detach()
 
    for epoch in range(args.epochs):
        #stoping=0
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        model.train()
        
        for i, (inp) in enumerate(train_loader):

            label=inp[2]
            label=np.array(label,dtype=np.int64)
            label=torch.from_numpy(label)

            label = label.to(device)

            optimizer.zero_grad() 

            output, final = model(all_drug_feat, edge_index, label_list,inp)

            log = torch.squeeze(output) 

            loss1 = loss_fct(log, label.long())


            loss_train = loss1


            loss_train.backward(retain_graph=False)

            optimizer.step()
           
            all_drug_feat = all_drug_feat.detach()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

            if i % 100 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))


        y_pred_train1 = []
        y_label_train = np.array(y_label_train)
        y_pred_train = np.array(y_pred_train).reshape((-1, 65))
        for i in range(y_pred_train.shape[0]):
            a = np.max(y_pred_train[i])
            for j in range(y_pred_train.shape[1]):
                if y_pred_train[i][j] == a:
                    
                    y_pred_train1.append(j)
                    break
        # y_pred_train1 = np.argmax(y_pred_train, axis=1)
        acc = accuracy_score(y_label_train, y_pred_train1)
        f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro', zero_division=0)
        recall1 = recall_score(y_label_train, y_pred_train1, average='macro', zero_division=0)
        precision1 = precision_score(y_label_train, y_pred_train1, average='macro', zero_division=0)

        print('epoch: {:04d}'.format(epoch + 1),
                        'loss_train: {:.4f}'.format(loss_train.item()),
                        'auroc_train: {:.4f}'.format(acc),
                        'f1_train: {:.4f}'.format(f1_score1),
                        'recall_train: {:.4f}'.format(recall1),
                        'precision_train: {:.4f}'.format(precision1),
                        'time: {:.4f}s'.format(time.time() - t))

        acc_test, f1_test, recall_test, precision_test, auc1, loss_test = test(model, val_loader, all_drug_feat, edge_index, label_list, args, 0, device)

        print('loss_val: {:.4f}'.format(loss_test.item()), 'acc_val: {:.4f}'.format(acc_test),
              'f1_val: {:.4f}'.format(f1_test), 'recall_val: {:.4f}'.format(recall_test),
              'precision_val: {:.4f}'.format(precision_test), 'roc_auc_val:{:.4f}'.format(auc1),'time: {:.4f}s'.format(time.time() - t))
        if f1_test > max_t1 + 1e-4:  # acc_val >= max_auc1 and f1_val>=max_f1:
            model_max = copy.deepcopy(model)
            all_drug_feat_best = all_drug_feat.detach().clone()
            max_auc1 = acc_test
            max_t1 = f1_test
            max_recall1 = recall_test
            max_precision1 = precision_test

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()


    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # print("valid:max_auc:{:.4f}".format(max_auc),'max_f1::{:.4f}'.format(max_f1),
    #       'recall_val: {:.4f}'.format(max_recall),'precision_val: {:.4f}'.format(max_precision))

    print("test:max_auc:{:.4f}".format(max_auc1), 'max_f1::{:.4f}'.format(max_t1),
          'recall_val: {:.4f}'.format(max_recall1), 'precision_val: {:.4f}'.format(max_precision1))

    acc_test, f1_test, recall_test,precision_test,auc1,loss_test = test(model_max, test_loader,all_drug_feat_best, edge_index,label_list, args, 1, device)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'acc_test: {:.4f}'.format(acc_test),
          'f1_test: {:.4f}'.format(f1_test), 'recall_test: {:.4f}'.format(recall_test),'precision_test: {:.4f}'.format(precision_test),'roc_auc_test:{:.4f}'.format(auc1))



def test(model, loader, all_drug_feat, edge_index, label_list, args, printfou, device):

    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.CrossEntropyLoss()
    # b_xent = nn.BCEWithLogitsLoss()
    model.eval()
    y_pred = []
    y_label = []
    # lbl = data_a.y
    zhongzi= args.zhongzi
    with torch.no_grad():
        for i, (inp) in enumerate(loader):
            label = inp[2]
            label = np.array(label, dtype=np.int64)
            label = torch.from_numpy(label)
            label = label.to(device)

            output,_ = model(all_drug_feat,edge_index,label_list, inp)
            log = torch.squeeze(m(output))

            loss1 = loss_fct(log, label.long())

            loss = loss1

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()

    y_pred_train1=[]
    y_label_train = np.array(y_label)
    y_pred_train = np.array(y_pred).reshape((-1, 65))
    for i in range(y_pred_train.shape[0]):
        a = np.max(y_pred_train[i])
        for j in range(y_pred_train.shape[1]):
            if y_pred_train[i][j] == a:
                y_pred_train1.append(j)
                break
    acc = accuracy_score(y_label_train, y_pred_train1)
    f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
    recall1 = recall_score(y_label_train, y_pred_train1, average='macro', zero_division=0)
    precision1 = precision_score(y_label_train, y_pred_train1, average='macro',zero_division=0)
    # auc2 = roc_auc_score(np.array(y_label_train).ravel(), np.array(y_pred_train1).ravel(), average='micro')


    y_label_train1 = np.zeros((y_label_train.shape[0], 65))
    for i in range(y_label_train.shape[0]):
        y_label_train1[i][y_label_train[i]] = 1

    auc_hong=0
    aupr_hong=0
    nn1 = y_label_train1.shape[1]
    for i in range(y_label_train1.shape[1]):

        if np.sum(y_label_train1[:, i].reshape((-1))) < 1:
            nn1 = nn1 - 1
            continue
        else:

            auc_hong = auc_hong + roc_auc_score(y_label_train1[:, i].reshape((-1)), y_pred_train[:, i].reshape((-1)))
            precision, recall, _thresholds = precision_recall_curve(y_label_train1[:, i].reshape((-1)),
                                                                    y_pred_train[:, i].reshape((-1)))
            aupr_hong = aupr_hong + auc(recall, precision)

    auc_macro = auc_hong / nn1
    aupr_macro = aupr_hong / nn1
    auc1 = roc_auc_score(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)), average='micro')
    precision, recall, _thresholds = precision_recall_curve(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)))
    aupr = auc(recall, precision)

    if printfou==1:
        with open(args.out_file, 'a') as f:

            f.write(str(zhongzi)+'  '+str(acc)+'   '+str(f1_score1)+'   '+str(recall1)+'   '+str(precision1)+'   '+str(auc1)+'   '+str(aupr)+'   '+str(auc_macro)+'   '+str(aupr_macro)+'\n')

    return acc,f1_score1,recall1,precision1,auc1,loss

