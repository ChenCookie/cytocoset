import os
import sys
import math
import time
import shutil
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

from model import CytoSetModel
from data import CytoDatasetFromFCS
from utils import (
    EarlyStopping, load_fcs_dataset, train_valid_split, combine_samples, down_rsampling
)


def test_model(test_samples, test_phenotypes, test_phenotypes_age, model, device):
    model.eval()
    correct_num, total_num = 0, 0
    y_pred, y_true = [], []
    a_pred, a_true = [], []
    losses = []
    ckpt_list = []
    tool_ckpt=[]
    count=0

    for sample, label ,age in zip(test_samples, test_phenotypes, test_phenotypes_age):
        with torch.no_grad():
            sample = torch.from_numpy(sample).to(device)
            true_label = torch.tensor([label], dtype=torch.float32).to(device)
            prob, pred_age,get_ckpt,alpha_percentage,same_pertentage,diff_percentage= model(sample)
            ckpt_list.append(get_ckpt.detach().cpu().numpy())
            pred_age = pred_age.int()

            loss = F.binary_cross_entropy(prob, true_label, reduction='mean')
            pred_label = torch.ge(prob, 0.5)

        losses.append(loss.item())
        v = (pred_label == label).sum()

        y_true.append(label)
        y_pred.append(prob.detach().cpu().numpy())
        a_true.append(age)
        a_pred.append(pred_age.detach().cpu().numpy())
        tool_ckpt.append(get_ckpt.detach().cpu().numpy())
        count+=1
        if count%10000==0:
            print(count)

        correct_num += v.item()
        total_num += 1

    acc = float(correct_num) / total_num

    y_true, y_pred = np.array(y_true), np.hstack(y_pred)
    auc = roc_auc_score(y_true, y_pred)
    eval_loss = np.mean(np.array(losses))

    # embedded_vector = np.empty((0,len(ckpt_list[0][0])), float)
    # for i_ckpt in range(0,len(ckpt_list),1):
    #     if i_ckpt%10000==0:
    #         print(i_ckpt)
    #     embedded_vector = np.append(embedded_vector, [np.array(ckpt_list[i_ckpt][0])], axis=0)
    print("start")
    embd_dict={'id':np.arange(0,len(a_true)),'sample_embd':ckpt_list,'label':y_true,'age': a_true}
    # df_embd = pd.DataFrame(embd_dict)
    # df_embd.to_csv('/playpen-ssd/chijane/cytoset/checkpoints/cd8_fuinf_GC/embd_autotune_median.csv')
    print("mid-1")
    f = open('/playpen-ssd/chijane/cytoset/checkpoints/cd8_fuinf_GC/embd_autotune_median.pkl', 'wb')
    print("mid-2")
    pickle.dump(embd_dict, f)
    print("mid-3")
    f.close()
    print("end")


    return eval_loss, acc, auc


def test(args):
    # load the pretrained model
    model = CytoSetModel.from_pretrained(model_path=args.model, config_path=args.config_file, cache_dir='./cache')
    model = model.to(args.device)

    # read the test dataset
    with open(args.test_pkl, 'rb') as f:
        test_data = pickle.load(f)
    test_samples, test_phenotypes,test_phenotypes_age = test_data['test_sample'], test_data['test_phenotype'], test_data['test_phenotype_age']

    # test model
    _, test_acc, test_auc = test_model(test_samples, test_phenotypes, test_phenotypes_age, model, args.device)
    # _, test_acc, test_auc = test_model(test_samples, test_phenotypes, model, args.device)
    print("Testing Acc: {:.3f}, Testing Auc: {:.3f}".format(test_acc, test_auc))

    # Finished the testing process
    print("Testing finished, Done")


def main():
    parser = argparse.ArgumentParser("Cytometry Set Model")

    # data
    parser.add_argument('--test_pkl', type=str, help='path or url to the test pickled file')

    # model
    parser.add_argument('--model', type=str, help='the path to the pretrained model')
    parser.add_argument('--config_file', type=str, help='the path to model configuration')
    parser.add_argument('--device', type=str, default='cuda', help='device to use')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'

    # test the model
    test(args)


if __name__ == "__main__":
    main()
