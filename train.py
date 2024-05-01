import os
import sys
import gc
import math
import time
import shutil
import argparse
from tqdm import tqdm
import numpy as np
import pickle

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA

import random
from meters import AverageMeter
from logger import WandbLogger
import pandas as pd

from model import CytoSetModel
from model import Config, count_params
from data import CytoDatasetFromFCS
from utils import (
    EarlyStopping, load_fcs_dataset, train_valid_split, combine_samples, down_rsampling
)

from sklearn.mixture import GaussianMixture

# for triplet
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier

random_state = np.random.RandomState(0)

clf_label = RandomForestClassifier(random_state=random_state)
clf_age = RandomForestClassifier(random_state=random_state)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def tune_percent3(precentage_weight,args_device,percent_scale,base_value, set_num_base=100):
    max_a, ids = torch.max(precentage_weight, 1, keepdim=True)
    precentage_weight=torch.scatter(precentage_weight,1, ids, set_num_base)
    set_range=torch.arange(1,percent_scale+1)/10*base_value
    set_range=set_range.to(args_device.device)
    precentage_weight=torch.sum(precentage_weight/set_num_base*set_range,1)
    precentage_weight=torch.round(precentage_weight*10)
    precentage_weight=torch.mode(precentage_weight)[0]/10
    return precentage_weight



# input triplet data
def load_triplet(root,number_base,triplet_filenmae,filename,label_file,percentage):
    filenamelist=[]
    filenameage=[]
    triplet_id = np.empty((0,3), int)
    triplet_age = np.empty((0,3), int)
    label_data=pd.read_csv(os.path.join(root, label_file))#all?(old: cla)
    label_ls=list(label_data['file_id'])#all?(old: cla)
    for line in open(os.path.join(root, filename)):
        single_file=line.rstrip('\n')
        single_file=single_file.split("/")[1]
        # filenamelist.append(single_file[5:10])#pree
        # filenameage.append(single_file[12:14])#pree
        # filenamelist.append(single_file.split('.')[0]) #preterm
        # label_idx=label_ls.index(single_file) #preterm
        # filenameage.append(label_data[triplet_filenmae.split('_')[0]][label_idx]) #preterm
        filenamelist.append(single_file.split('_')[1]) #lung
        label_idx=label_ls.index(single_file) #lung
        filenameage.append(label_data[triplet_filenmae.split('_')[0]][label_idx]) #lung

    for line in open(os.path.join(root,'tripletlists_maxDrugRelatedAE_'+number_base, triplet_filenmae)): # change end name
        single_triplet=line.rstrip('\n')
        single_triplet=single_triplet.split(' ')
        id_string=[np.array(filenamelist)[int(single_triplet[0])],np.array(filenamelist)[int(single_triplet[1])],np.array(filenamelist)[int(single_triplet[2])]]
        age_string=[np.array(filenameage)[int(single_triplet[0])],np.array(filenameage)[int(single_triplet[1])],np.array(filenameage)[int(single_triplet[2])]]
        triplet_id = np.append(triplet_id, [np.array(id_string)], axis=0)
        triplet_age = np.append(triplet_age, [np.array(age_string)], axis=0)
    sampled_list = random.sample(range(0,len(triplet_id),1), int(len(triplet_id)*percentage))
    triplet_id=triplet_id[sampled_list]
    triplet_age=triplet_age[sampled_list]
    return triplet_id,triplet_age

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def accuracy(dist_a, dist_b):
    margin = 0

    pred = (dist_a - dist_b - margin).cpu().data
    return (pred > 0).sum()*1.0/dist_a.size()[0]


def test_valid(test_loader, model, args):
    """ Test the model performance """
    model.eval()
    losses = AverageMeter(round=3)
    correct_num, total_num,svm_correct_num, svm_total_num,svm_avg_mse = 0, 0, 0, 0, 0
    y_pred, y_true = [], []
    a_pred, a_true = [], []
    

    val_embd_concatenate=np.zeros((0,args.h_dim), dtype=float)
    val_true_label_concatenate=np.zeros((0,), dtype=int)
    val_true_age_concatenate=np.zeros((0,), dtype=int)
    save_loss=dict()

    for x, y, a, sid in test_loader:
        x, y , a = x.to(args.device), y.to(args.device), a.to(args.device)
        
        with torch.no_grad():
            prob, pred_age,get_ckpt,alpha_percentage,same_percentage,diff_percentage = model(x)
            alpha_percentage=tune_percent3(alpha_percentage,args,9,1)
            same_percentage=tune_percent3(same_percentage,args,5,2)
            diff_percentage=tune_percent3(diff_percentage,args,5,2)



            triplet_smaple,triplet_label=[],[]

            for set_percentage in range(1,11,1):
                sub_triplet_sample, sub_triplet_age=load_triplet('/playpen-ssd/chijane/data_lung/data_lung_1',str(int(args.name_end)),'DrugRelatedAE_tripletlist_subpick_trainval_rffmax_same'+str(format(same_percentage.item(),'.1f'))+'_diff'+str(format(diff_percentage.item(),'.1f'))+'.txt','filenames_'+str(int(args.name_end))+'.json','fcs_info.csv',set_percentage/10)
                triplet_smaple.append(sub_triplet_sample)
                triplet_label.append(sub_triplet_age)


            triplet_table=[]
            for per_triplet_smaple in triplet_smaple:
                checking_table = np.empty((0,3), int)
                for single_triplet in per_triplet_smaple:
                    check_bool=[]
                    for triplet_id in single_triplet:
                        if int(triplet_id) in sid.tolist():
                            check_bool.append(1)
                        else:
                            check_bool.append(0)
                    checking_table = np.append(checking_table, [np.array(check_bool)], axis=0)
                triplet_table.append(checking_table)


            sid_arr=sid.detach().cpu().numpy()
            a_arr=a.detach().cpu().numpy()
            y_arr=y.detach().cpu().numpy()
            ckpt_arr=get_ckpt.detach().cpu().numpy()

            for set_percentage in range(1,11,1):
                data1_id_lst=list()
                data2_id_lst=list()
                data3_id_lst=list()

                get_triplst=triplet_smaple[set_percentage-1]
                age_triplst=triplet_label[set_percentage-1]
                checking_table=triplet_table[set_percentage-1]

                for single_lst in range(0,len(checking_table),1):
                    if sum(checking_table[single_lst])==3:
                        data1_idx=intersection(np.where(sid_arr==int(get_triplst[single_lst][0]))[0],np.where(a_arr==int(age_triplst[single_lst][0]))[0])
                        data2_idx=intersection(np.where(sid_arr==int(get_triplst[single_lst][1]))[0],np.where(a_arr==int(age_triplst[single_lst][1]))[0])
                        data3_idx=intersection(np.where(sid_arr==int(get_triplst[single_lst][2]))[0],np.where(a_arr==int(age_triplst[single_lst][2]))[0])
                        if len(data1_idx)>0 and len(data2_idx)>0 and len(data3_idx)>0:
                            data1_id_lst.append(random.choice(data1_idx))
                            data2_id_lst.append(random.choice(data2_idx))
                            data3_id_lst.append(random.choice(data3_idx))

                    
                if len(data1_id_lst)>0 and len(data2_id_lst)>0 and len(data3_id_lst)>0:
                    data1=get_ckpt[data1_id_lst]
                    data2=get_ckpt[data2_id_lst]
                    data3=get_ckpt[data3_id_lst]
                    dista = F.pairwise_distance(data1, data2, 2)
                    distb = F.pairwise_distance(data1, data3, 2)
                    target = torch.FloatTensor(dista.size()).fill_(1)
                    if torch.cuda.is_available():
                        target = target.cuda()
                    target = Variable(target)
                    result=F.margin_ranking_loss(dista, distb, target,margin=0.2, reduction = 'mean')
                    if set_percentage-1 not in save_loss.keys():
                        save_loss[set_percentage-1]=[result]
                    else:
                        get_loss_lst=save_loss[set_percentage-1]
                        get_loss_lst.append(result)
                        save_loss[set_percentage-1]=get_loss_lst

            val_embd_concatenate=np.concatenate((val_embd_concatenate, ckpt_arr), axis=0)
            val_true_label_concatenate=np.concatenate((val_true_label_concatenate, y_arr), axis=0)
            val_true_age_concatenate=np.concatenate((val_true_age_concatenate, a_arr), axis=0)
            
            pred_age = pred_age.int()


            loss= (alpha_percentage) * F.binary_cross_entropy(prob, y, reduction='mean')+(1-alpha_percentage)*F.margin_ranking_loss(dista, distb, target,margin=0.2, reduction = 'mean')
            pred_label = torch.ge(prob, 0.5)

            

        losses.update(loss.item(), n=x.size(0))
        v = (pred_label == y).sum()

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(prob.detach().cpu().numpy())
        a_true.append(a.detach().cpu().numpy())
        a_pred.append(pred_age.detach().cpu().numpy())

        correct_num += v.item()
        total_num += x.size(0)


    predict_label=clf_label.predict(np.array(val_embd_concatenate))
    predict_age=clf_age.predict(np.array(val_embd_concatenate))
    for i in save_loss.keys():
        # loss_list=save_loss[i]
        save_loss[i]=(sum(save_loss[i]))/len(save_loss[i]) 
    if len(save_loss.keys())==0:
        args.triplet_percent=(random.randint(0,9)+1)/10
    else:
        args.triplet_percent=(min(save_loss, key=save_loss.get)+1)/10
    args.triplet_percent=1
 

    print("test valid predict/true:",predict_label,val_true_label_concatenate)
    num_correct = sum(p == t for p, t in zip(predict_label, val_true_label_concatenate))
    print("test_valid SVM accuracy:",num_correct/len(predict_label),num_correct,len(predict_label))
    
    svm_mse=sum(p == t for p, t in zip(val_true_age_concatenate, predict_age))
    print("test MSE valid predict/true:",predict_age,val_true_age_concatenate)
    print("test_valid MSE accuracy:",svm_mse/len(predict_label),svm_mse,len(predict_label))
    print("in test_valid alpha=",format(alpha_percentage.item(),'.1f'))
    
    svm_correct_num+=num_correct
    svm_total_num+=len(predict_label)
    svm_avg_mse+=svm_mse

    acc = float(correct_num) / total_num

    y_true, y_pred = np.hstack(y_true), np.hstack(y_pred)
    y_pred_trans = np.copy(y_pred)
    y_pred_trans[y_pred_trans==2]=0

    y_true_trans = np.copy(y_true)
    y_true_trans[y_true_trans==2]=0

    auc = roc_auc_score(y_true_trans, y_pred_trans)
    
    a_true, a_pred = np.hstack(a_true), np.hstack(a_pred)
    age_binary=sum(p == t for p, t in zip(a_pred, a_true))/len(a_true)


    return acc, losses.avg, auc,age_binary,svm_correct_num/svm_total_num,svm_avg_mse/svm_total_num


def test_model(test_samples, test_phenotypes, test_phenotypes_age, model, device,triplet_percent,seed_pick,alpha_pick,same_pick,diff_pick,name_end_pick):
    model.eval()
    correct_num, total_num = 0, 0
    y_pred, y_true = [], []
    a_pred, a_true = [], []
    losses = []
    ckpt_list = []

    get_triplst, age_triplst=load_triplet('/playpen-ssd/chijane/data_lung/data_lung_1',str(int(name_end_pick)),'DrugRelatedAE_tripletlist_subpick_test_rffmax_same'+str(same_pick)+'_diff'+str(diff_pick)+'.txt','filenames_'+str(int(name_end_pick))+'.json','fcs_info.csv',1)

    sid=pd.read_csv('/playpen-ssd/chijane/data_lung/data_lung_1/lung_fcs/all/test_labels_'+str(int(name_end_pick))+'.csv')
    # sid=[int(one_sid.split('_')[1][0:5]) for one_sid in sid['fcs_filename']] # pree
    # sid=[int(one_sid.split('.')[0]) for one_sid in sid['csv_filename']] #preterm
    # sid=[int(one_sid.split('.')[0]) for one_sid in sid['fcs_filename']] #covid
    sid=[int(one_sid.split('_')[1]) for one_sid in sid['fcs_filename']] #lung
    tool_ckpt=[]

    for sample, label ,age in zip(test_samples, test_phenotypes, test_phenotypes_age):
        with torch.no_grad():
            sample = torch.from_numpy(sample).to(device)
            true_label = torch.tensor([label], dtype=torch.float32).to(device)
            prob, pred_age,get_ckpt,alpha_percentage,same_pertentage,diff_percentage= model(sample)
            ckpt_list.append(get_ckpt.detach().cpu().numpy())
            pred_age = pred_age.int()


            loss = (alpha_pick) * F.binary_cross_entropy(prob, true_label, reduction='mean')
            pred_label = torch.ge(prob, 0.5)

        losses.append(loss.item())
        v = (pred_label == label).sum()

        y_true.append(label)
        y_pred.append(prob.detach().cpu().numpy())
        a_true.append(age)
        a_pred.append(pred_age.detach().cpu().numpy())
        tool_ckpt.append(get_ckpt.detach().cpu().numpy())

        correct_num += v.item()
        total_num += 1


    predict_label=clf_label.predict(np.array(tool_ckpt)[:,0,:])
    label_prob=clf_label.predict_proba(np.array(tool_ckpt)[:,0,:])
    label_prob = label_prob[:, 1]
    predict_age=clf_age.predict(np.array(tool_ckpt)[:,0,:])
    age_prob=clf_age.predict_proba(np.array(tool_ckpt)[:,0,:])
    age_prob = age_prob[:, 1]

    
    print("for alpha=",alpha_pick,seed_pick,triplet_percent,",same diff=",same_pick,diff_pick)

    predict_label = [1 if single_label > 0 else 0 for single_label in predict_label]


    
    num_correct = sum(p == t for p, t in zip(predict_label, np.array(y_true)))
    svm_acc=num_correct/len(predict_label)
    svm_mse= sum(p == t for p, t in zip(predict_age, np.array(a_true)))/len(predict_age)


    checking_table = np.empty((0,3), int)
    for single_triplet in get_triplst:
        check_bool=[]
        for triplet_id in single_triplet:
            if int(triplet_id) in sid:
                check_bool.append(1)
            else:
                check_bool.append(0)
        checking_table = np.append(checking_table, [np.array(check_bool)], axis=0)


    sid_arr=np.array(sid)
    a_arr=np.array(a_true)
    ckpt_arr=np.array(tool_ckpt)
    data1= np.empty((0,1,256), float)
    data2= np.empty((0,1,256), float)
    data3= np.empty((0,1,256), float)
    for single_lst in range(0,len(checking_table),1):
        if sum(checking_table[single_lst])==3:
            data1_idx=intersection(np.where(sid_arr==int(get_triplst[single_lst][0]))[0],np.where(a_arr==int(age_triplst[single_lst][0]))[0])
            data2_idx=intersection(np.where(sid_arr==int(get_triplst[single_lst][1]))[0],np.where(a_arr==int(age_triplst[single_lst][1]))[0])
            data3_idx=intersection(np.where(sid_arr==int(get_triplst[single_lst][2]))[0],np.where(a_arr==int(age_triplst[single_lst][2]))[0])
            if len(data1_idx)>0 and len(data2_idx)>0 and len(data3_idx)>0:
                data1_idx=random.choice(data1_idx)
                data2_idx=random.choice(data2_idx)
                data3_idx=random.choice(data3_idx)
                data1=np.append(data1, [ckpt_arr[data1_idx]], axis=0)
                data2=np.append(data2, [ckpt_arr[data2_idx]], axis=0)
                data3=np.append(data3, [ckpt_arr[data3_idx]], axis=0)
    data1=torch.from_numpy(data1)
    data2=torch.from_numpy(data2)
    data3=torch.from_numpy(data3)
    data1, data2, data3 = data1.to(device), data2.to(device), data3.to(device)
    dista = F.pairwise_distance(data1, data2, 2)
    distb = F.pairwise_distance(data1, data3, 2)

    target = torch.FloatTensor(dista.size()).fill_(1)
    if torch.cuda.is_available():
        target = target.cuda()
    target = Variable(target)
    loss_triplet=F.margin_ranking_loss(dista, distb, target,margin=0.2)




    acc_triplet = accuracy(dista, distb)
    
    acc = float(correct_num) / total_num

    y_true, y_pred = np.array(y_true), np.hstack(y_pred)
    a_true, a_pred = np.hstack(a_true), np.hstack(a_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)

    y_pred_trans = np.copy(y_pred)
    y_pred_trans[y_pred_trans==2]=0

    y_true_trans = np.copy(y_true)
    y_true_trans[y_true_trans==2]=0

    auc = roc_auc_score(y_true_trans, y_pred_trans)# pree y_true y_pred
    age_mse=F.mse_loss(torch.Tensor(a_pred), torch.Tensor(a_true), reduction='mean')
    print("test MSE",age_mse)
    eval_loss = np.mean(np.array(losses))

    dist_diff_list= []
    age_diff_list= []
    embedded_vector = np.empty((0,len(ckpt_list[0][0])), float)
    for i_ckpt in range(0,len(ckpt_list),1):
        embedded_vector = np.append(embedded_vector, [np.array(ckpt_list[i_ckpt][0])], axis=0)
        for j_ckpt in range(i_ckpt+1,len(ckpt_list),1):
            age_diff_list.append(abs(a_true[i_ckpt]-a_true[j_ckpt]))
            ssum=float(torch.sqrt(sum(torch.flatten((torch.from_numpy(ckpt_list[i_ckpt] - ckpt_list[j_ckpt])).pow(2)))))
            dist_diff_list.append(ssum)


    embd_dict={'id':np.arange(0,len(a_true)),'sample_embd':list(embedded_vector),'label':y_true,'label_predict':predict_label,'label_prob':label_prob,'age': a_true,'age_predict':predict_age,'age_prob':age_prob}
    df_embd = pd.DataFrame(embd_dict)
    df_embd.to_csv('/playpen-ssd/chijane/cytoset/scripts/train/newlung_result/trail'+str(int(name_end_pick))+'/embd_balence_goodmax_alpha'+str(format(alpha_pick.item(),'.1f'))+'_'+str(seed_pick)+'same'+str(same_pick)+'_diff'+str(diff_pick)+'_forDrugRelatedAEmax.csv')


    print("loss triplet: ",loss_triplet)
    print("accuracy triplet: ",acc_triplet)
    print("svm acc auc: ",svm_acc, roc_auc_score(y_true_trans, label_prob))
    print("svm mse: ",svm_mse)

    return eval_loss, acc, auc, fpr, tpr, age_mse.numpy()


def train(args):
    set_seed(args.seed)

    logger = WandbLogger(
        logger_name=f'CytoSet-{args.ncell}@{args.pool}',
        log_dir=args.log_dir,
        stream=sys.stdout,
        args=args,
        wandb_project='CytoSet'
    )

    # set model
    model = CytoSetModel(args).to(args.device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.wts_decay
    )

    if args.ckpt is not None:
        print(f'Loading model from {args.ckpt}')
        checkpoint = torch.load(args.ckpt, map_location='cpu' if not torch.cuda.is_available() else None)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])

    # set dataloader
    if args.pkl:
        with open(args.train_pkl, 'rb') as f:
            _data = pickle.load(f)
            train_samples, train_phenotypes = _data['sample'], _data['phenotype']
        with open(args.test_pkl, 'rb') as f:
            _data = pickle.load(f)
            test_samples, test_phenotypes = _data['sample'], _data['phenotype']
    else:
        train_samples, train_phenotypes, train_phenotypes_age = load_fcs_dataset(
            args.train_fcs_info, args.markerfile, args.co_factor
        )

        test_samples, test_phenotypes, test_phenotypes_age = load_fcs_dataset(
            args.test_fcs_info, args.markerfile, args.co_factor
        )

    valid_phenotypes = train_phenotypes
    valid_phenotypes_age = train_phenotypes_age
    sample_id_list=np.array(pd.read_csv(args.train_fcs_info, sep=','))[:, 0]
    train_samples_list=list()
    for single_id in sample_id_list:
        # raw_id=single_id.split(".") #preterm
        # train_samples_list.append(raw_id[0]) #preterm
        # raw_id=single_id.split("_")# pree
        # train_samples_list.append(raw_id[1][0:5]) # pree
        raw_id=single_id.split("_")# lung
        train_samples_list.append(raw_id[1]) # lung
    train_samples_array=np.array(train_samples_list)
    train_samples_array=train_samples_array.astype(np.int)

    if (args.valid_fcs_info is not None or args.valid_pkl is not None) or args.generate_valid:
        if args.valid_fcs_info is not None:
            valid_samples, valid_phenotypes, valid_phenotypes_age = load_fcs_dataset(
                args.valid_fcs_info, args.markerfile, args.co_factor
            )
            X_train, id_train = combine_samples(train_samples, np.arange(len(train_samples)))
            X_valid, id_valid = combine_samples(valid_samples, np.arange(len(valid_samples)))
        elif args.valid_pkl is not None:
            with open(args.valid_pkl, 'rb') as f:
                _valid = pickle.load(f)
            valid_samples, valid_phenotypes = _valid['sample'], _valid['phenotype']
            X_train, id_train = combine_samples(train_samples, np.arange(len(train_samples)))
            X_valid, id_valid = combine_samples(valid_samples, np.arange(len(valid_samples)))
        else:
            X_train, id_train, X_valid, id_valid = train_valid_split(
                train_samples, np.arange(len(train_samples))
            )

        del train_samples
   
        gc.collect()
        X_train, id_train = shuffle(X_train, id_train)
        train_data = CytoDatasetFromFCS(X_train, train_samples_array, id_train, train_phenotypes, train_phenotypes_age,
                                        args.ncell, args.nsubset, args.per_sample)#id_train to train_samples_array[id_train] cla: X_train, train_samples_array, id_train, train_phenotypes, train_phenotypes_age,args.ncell, args.nsubset, args.per_sample
        valid_data = CytoDatasetFromFCS(X_valid, train_samples_array, id_valid, valid_phenotypes, valid_phenotypes_age,
                                        args.ncell, args.nsubset, args.per_sample)#id_valid to train_samples_array[id_valid]
    else:
        X_train, id_train = combine_samples(train_samples, np.arange(len(train_samples)))
        del train_samples
        gc.collect()

        X_train, id_train = shuffle(X_train, id_train)
        train_data = CytoDatasetFromFCS(X_train,train_samples_array, id_train, train_phenotypes,train_phenotypes_age,
                                        args.ncell, args.nsubset, args.per_sample)
        logger.info("Neither having valid dataset nor generating valid dataset, use train data as valid dataset")
        valid_data = CytoDatasetFromFCS(X_train,train_samples_array, id_train, train_phenotypes,train_phenotypes_age,
                                        args.ncell, args.nsubset, args.per_sample)
    print("after loaddd")
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=1,
        drop_last=True,
        pin_memory=False
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )

    logger.info('**** Start Training ****')
    logger.info(f' config: {args.ncell}@{args.pool}')
    logger.info(f' Total epochs: {args.n_epochs}')
    logger.info('Total Params: {:.2f}M'.format(count_params(model) / 1e6))

    losses = AverageMeter(round=3)
    data_time = AverageMeter(round=3)
    step_time = AverageMeter(round=3)


    best_auc = 0
    best_svm_acc,best_svm_mse=0,0
    pbar = tqdm(range(args.n_epochs), initial=0, dynamic_ncols=True, smoothing=0.01)



    # start the main training loop
    for epoch in pbar:
        model.train()
        embd_concatenate=np.zeros((0,args.h_dim), dtype=float)
        true_label_concatenate=np.zeros((0,), dtype=int)
        true_age_concatenate=np.zeros((0,), dtype=int)
        

        # get the data
        for x, y, a, sid in train_loader:
            start_time = time.time()
            x, y, a, sid = x.to(args.device), y.to(args.device), a.to(args.device), sid.to(args.device)
            # count data moving time
            data_time.update(time.time() - start_time)


            # model feed forward
            prob, pred_age,get_ckpt,alpha_percentage,same_percentage,diff_percentage= model(x)

            alpha_percentage=tune_percent3(alpha_percentage,args,9,1)
            same_percentage=tune_percent3(same_percentage,args,5,2)
            diff_percentage=tune_percent3(diff_percentage,args,5,2)

            # input triplet data
            get_triplst, age_triplst=load_triplet('/playpen-ssd/chijane/data_lung/data_lung_1',str(int(args.name_end)),'DrugRelatedAE_tripletlist_subpick_trainval_rffmax_same'+str(format(same_percentage.item(),'.1f'))+'_diff'+str(format(diff_percentage.item(),'.1f'))+'.txt','filenames_'+str(int(args.name_end))+'.json','fcs_info.csv',args.triplet_percent)


            checking_table = np.empty((0,3), int)
            for single_triplet in get_triplst:
                check_bool=[]
                for triplet_id in single_triplet:
                    if int(triplet_id) in sid.tolist():
                        check_bool.append(1)
                    else:
                        check_bool.append(0)
                checking_table = np.append(checking_table, [np.array(check_bool)], axis=0)
            triplet_count=0
            sid_arr=sid.detach().cpu().numpy()
            a_arr=a.detach().cpu().numpy()
            y_arr=y.detach().cpu().numpy()
            ckpt_arr=get_ckpt.detach().cpu().numpy()
            data1_id_lst=list()
            data2_id_lst=list()
            data3_id_lst=list()

            for single_lst in range(0,len(checking_table),1):
                if sum(checking_table[single_lst])==3:
                    triplet_count+=1
                    data1_idx=intersection(np.where(sid_arr==int(get_triplst[single_lst][0]))[0],np.where(a_arr==int(age_triplst[single_lst][0]))[0])
                    data2_idx=intersection(np.where(sid_arr==int(get_triplst[single_lst][1]))[0],np.where(a_arr==int(age_triplst[single_lst][1]))[0])
                    data3_idx=intersection(np.where(sid_arr==int(get_triplst[single_lst][2]))[0],np.where(a_arr==int(age_triplst[single_lst][2]))[0])

                    
                    if len(data1_idx)>0 and len(data2_idx)>0 and len(data3_idx)>0:
                        data1_id_lst.append(random.choice(data1_idx))
                        data2_id_lst.append(random.choice(data2_idx))
                        data3_id_lst.append(random.choice(data3_idx))


            data1=get_ckpt[data1_id_lst]
            data2=get_ckpt[data2_id_lst]
            data3=get_ckpt[data3_id_lst]

            dista = F.pairwise_distance(data1, data2, 2)
            distb = F.pairwise_distance(data1, data3, 2)

            target = torch.FloatTensor(dista.size()).fill_(1)
            if torch.cuda.is_available():
                target = target.cuda()
            target = Variable(target)


            embd_concatenate=np.concatenate((embd_concatenate, ckpt_arr), axis=0) #ckpt_arr
            true_label_concatenate=np.concatenate((true_label_concatenate, y_arr), axis=0)
            true_age_concatenate=np.concatenate((true_age_concatenate, a_arr), axis=0)

            
            pred_age = pred_age.int()
            a_int=a.int()
            
            th_bool=torch.eq(pred_age, a_int)

            loss= (alpha_percentage) * F.binary_cross_entropy(prob, y, reduction='mean')+(1-alpha_percentage) *F.margin_ranking_loss(dista, distb, target,margin = 0.2, reduction = 'mean')

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_time.update(time.time() - start_time)
            losses.update(loss.item())

        clf_label.fit(embd_concatenate, true_label_concatenate)
        clf_age.fit(embd_concatenate, true_age_concatenate)
        print("train alpha=",alpha_percentage,"same=",same_percentage,"diff=",diff_percentage,"triplet_percent=",args.triplet_percent)
        print("embd_concatenate\n",embd_concatenate)

        # log the training progress
        if (epoch + 1) % args.log_interval == 0:
            val_acc, val_loss, val_auc, val_mse,get_svm_acc, get_svm_mse = test_valid(valid_loader, model, args)

            pbar.set_description(
                "Epoch: {}/{}, data: {:.3f}, step: {:.3f}, loss: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}, val_auc: {:.3f}, val_mse: {:.3f}".format(
                    str(epoch + 1).zfill(4), args.n_epochs, data_time.avg,
                    step_time.avg, losses.avg, val_loss, val_acc, val_auc, val_mse
                )
            )

            stats = {
                'epoch': epoch + 1,
                'loss': losses.avg,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_mse': val_mse
            }
            logger._log_to_wandb(stats=stats, epoch=epoch + 1)

            # check early stop condition
            early_stopping(val_loss=val_loss)
            if early_stopping.early_stop:
                logger.info(f"Training early stops at epoch: {epoch+1}")
                break

        losses.reset()
        data_time.reset()
        step_time.reset()

        if (epoch + 1) % args.save_interval == 0: 
            val_acc, val_loss, val_auc, val_mse,get_svm_acc, get_svm_mse = test_valid(valid_loader, model, args) # ,condition_mask,age_mask ,train_condition,train_age ,store_train_embd,store_train_label,store_train_age
            is_best = val_auc >= best_auc
            best_auc = max(best_auc, val_auc)

            ckpt_file = f"{args.log_dir}/{str(epoch + 1).zfill(4)}.ckpt"

            torch.save(
                {
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': args,
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'val_mse': val_mse,
                    'epoch': epoch + 1
                },
                ckpt_file
            )
            if is_best:
                # save the best model and check points
                torch.save(model.state_dict(), f'{args.log_dir}/best_model.pt')
                shutil.copyfile(ckpt_file, f'{args.log_dir}/best.ckpt')
            
            print("Test SVM accuracy overall=",get_svm_acc)
            print("svm MSE: ",get_svm_mse)

        pbar.update()

    pbar.close()

    logger.info("**** Training Finished ****")

    # load best model and check the performance on test data
    # format test samples

    test_samples = [np.expand_dims(sample, 0).astype(np.float32) for sample in test_samples]
    if args.test_rsampling:
        test_samples = [down_rsampling(sample, args.ncell, axis=1) for sample in test_samples]
    state_dict = torch.load(f'{args.log_dir}/best_model.pt', map_location='cpu' if not torch.cuda.is_available else None)
    model.load_state_dict(state_dict, strict=True)
    print("before testset alpha=",format(alpha_percentage.item(),'.1f'))

    # test model
    _, test_acc, test_auc, test_fpr, test_tpr, test_mse = test_model(test_samples, test_phenotypes, test_phenotypes_age, model, args.device,args.triplet_percent,args.seed,alpha_percentage,format(same_percentage.item(),'.1f'),format(diff_percentage.item(),'.1f'),args.name_end) # ,train_condition,train_age ,store_train_embd,store_train_label,store_train_age
    logger.info("Testing Acc: {:.3f}, Testing Auc: {:.3f}".format(test_acc, test_auc))

    with open(f'{args.log_dir}/test_result.pkl', 'wb') as f:
        test_stat = {
            'test_sample': test_samples,
            'test_phenotype': test_phenotypes,
            'test_phenotype_age': test_phenotypes_age,
            'fpr': test_fpr,
            'tpr': test_tpr,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_mse': test_mse
        }
        pickle.dump(test_stat, f)

    # Finished the training and testing, saving the configurations
    logger.info("Testing finished, saving training configurations....")
    config = Config.from_args(args)
    config.to_json_file(f"{args.log_dir}/config.json")
    logger.info("Done")


def main():
    parser = argparse.ArgumentParser("Cytometry Set Model")

    # model
    parser.add_argument('--in_dim', default=37, type=int, help="input dim")
    parser.add_argument('--h_dim', default=64, type=int, help='hidden dims to use in the model')
    parser.add_argument('--pool', default='max', choices=['mean', 'max', 'sum'], type=str, help='block pooling type')
    parser.add_argument('--out_pool', default='mean', choices=['mean', 'max', 'sum'], type=str, help='output pooling type')
    parser.add_argument('--nblock', default=1, type=int, help="# of blocks to use in the model")
    parser.add_argument('--triplet_percent', default=1, type=float, help="triplet percentage")
    parser.add_argument('--alpha', default=0.5, type=float, help='percentage of cytoset')
    parser.add_argument('--same_t', default=0.2, type=float, help='Threhold of same')
    parser.add_argument('--diff_t', default=0.2, type=float, help='Threhold of diff')
    parser.add_argument('--name_end', default=None, type=float, help='train/test file end name')

    # optimizer
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta_1 params in the optimizer')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta_2 params in the optimizer')
    parser.add_argument('--wts_decay', default=1e-3, type=float, help='coefficient of weight decay')
    parser.add_argument('--patience', default=5, type=int, help='the patience param for early stopping')

    # data
    parser.add_argument('--train_fcs_info', type=str, default=None, help='path to train fcs info file')
    parser.add_argument('--valid_fcs_info', default=None, type=str, help='path to valid fcs info file')
    parser.add_argument('--test_fcs_info', type=str, default=None, help='path to test fcs info file')
    parser.add_argument('--train_pkl', type=str, default=None, help='path to the training pickle file')
    parser.add_argument('--valid_pkl', type=str, default=None, help='path to the valid pickle file')
    parser.add_argument('--test_pkl', type=str, default=None, help='path to the testing pickle file')

    parser.add_argument('--markerfile', type=str, help='path to marker indication file')
    parser.add_argument('--generate_valid', action='store_true', help='whether to generate valid data from train data')
    parser.add_argument('--test_rsampling', action='store_true', help='whether to test model using sampled data')
    parser.add_argument('--pkl', action='store_true', help='load data directly from pickled data')

    parser.add_argument('--batch_size', default=200, type=int, help='batch size of labeled data')
    parser.add_argument('--nsubset', default=1024, type=int, help='total number of multi-cell inputs that will be generated per class')
    parser.add_argument('--ncell', default=200, type=int, help='number of cells per multi-cell input')
    parser.add_argument('--co_factor', default=5, type=float, help='arcsinh normalization factor')
    parser.add_argument('--per_sample', action='store_true', help='whether the nsubset argument refers to each class or each input')

    parser.add_argument('--shuffle', action='store_true', help='whether to shuffle the data')
    parser.add_argument('--n_epochs', default=200, type=int, help='number of total training epochs')
    parser.add_argument('--log_dir', default='./exp', type=str, help='path to log dir')
    parser.add_argument('--log_interval', default=1, type=int, help='logging interval')
    parser.add_argument('--save_interval', default=5, type=int, help='save model interval')

    # utils
    parser.add_argument('--seed', default=12345, type=int, help='random seed to use')
    parser.add_argument('--device', default='cuda', type=str, help='specify the training device')
    parser.add_argument('--ckpt', default=None, type=str, help='path to the checkpoint file')

    args = parser.parse_args()


    if not torch.cuda.is_available():
        args.device = 'cpu'

    # train the model
    train(args) 


if __name__ == "__main__":
    main()
