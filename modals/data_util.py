import copy
import os
import random
from pathlib import Path

import dill
import torch
import torchtext.data as data
import torchtext.datasets as txtdatasets
from torch.utils.data import Sampler,Subset,DataLoader
from torchtext.vocab import GloVe
from modals.setup import EMB_DIR
from modals.datasets import EDFX,PTBXL,Chapman,WISDM,ICBEB,Georgia
from modals.operation_tseries import ECG_NOISE_DICT, ToTensor,RandAugment,TransfromAugment,TransfromAugment_classwise,InfoRAugment,BeatAugment
from sklearn.preprocessing import StandardScaler

def save_txt_dataset(dataset, path):
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(dataset.examples, path/"examples.pkl", pickle_module=dill)
    torch.save(dataset.fields, path/"fields.pkl", pickle_module=dill)


def load_txt_dataset(path, fields):
    if not isinstance(path, Path):
        path = Path(path)
    examples = torch.load(path/"examples.pkl", pickle_module=dill)
    # fields = torch.load(path/"fields.pkl", pickle_module=dill)
    return data.Dataset(examples, fields)


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


def binarize(dataset):
    binary_examples = []
    for example in dataset.examples:
        if example.label != 'neutral':
            binary_examples.append(example)
    dataset.examples = binary_examples
    return dataset

def get_image_dataloaders(dataset_name, valid_size=-1, batch_size=16, dataroot='.data',
                          augselect=None,randaug_dic={},fix_policy_list=[],class_wise=False,info_region=None,rd_seed=None,test_augment=False,
                          num_workers=8):
    #rand augment !!! have bug when not using default split
    train_transfrom = []
    class_wise_transfrom = []
    if randaug_dic.get('randaug',False):
        print('Using RandAugment')
        train_transfrom.extend([
            ToTensor(),
            RandAugment(randaug_dic['rand_n'],randaug_dic['rand_m'],rd_seed=rd_seed,augselect=randaug_dic['augselect'])])
    if len(fix_policy_list)>0:
        print('Using Transfrom')
        if class_wise:
            print('Class-Wise') #tmp fix of num_class!!!
            class_wise_transfrom.extend([
                ToTensor(),
                TransfromAugment_classwise(fix_policy_list,m=randaug_dic['rand_m'],n=randaug_dic['rand_n'],
                num_class=5,rd_seed=rd_seed,p=randaug_dic['aug_p']) #tmp!!!
            ])
        elif info_region!=None:
            print('Infomation Region')
            train_transfrom.extend([
                ToTensor(),
                InfoRAugment(fix_policy_list,m=randaug_dic['rand_m'],n=randaug_dic['rand_n'],
                mode=info_region,rd_seed=rd_seed,p=randaug_dic['aug_p'])
            ])
        else:
            train_transfrom.extend([
                ToTensor(),
                TransfromAugment(fix_policy_list,m=randaug_dic['rand_m'],n=randaug_dic['rand_n'],
                rd_seed=rd_seed,p=randaug_dic['aug_p'])
            ])
    print('Train transfrom: ',train_transfrom)
    if test_augment:
        print('Using valid/test transfrom, just for experiment')
        valid_transfrom = train_transfrom
        test_transfrom = train_transfrom
    else:
        valid_transfrom = []
        test_transfrom = []
    #choose dataset
    dataset_func = None
    if dataset_name == 'mimic_lt':
        train = dataset_func(dataroot,mode='train',augmentations=train_transfrom)
        valid = dataset_func(dataroot,mode='valid',augmentations=valid_transfrom)
        test = dataset_func(dataroot,mode='test',augmentations=test_transfrom)
        classes = ['negative', 'positive']
    else:
        ValueError(f'Invalid dataset name={dataset_name}')

    classes = [i for i in range(train.num_class)]
    input_channel = train.channel

    train_loader = DataLoader(train,batch_size=batch_size, shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=True)
    valid_loader = DataLoader(valid,batch_size=batch_size, shuffle=False,num_workers=num_workers,pin_memory=True)
    test_loader = DataLoader(test,batch_size=batch_size, shuffle=False,num_workers=num_workers,pin_memory=True)

    print('### Dataset ###')
    print(f'=>{dataset_name}')
    print(f'  |Train size:\t{len(train)}')
    if valid is not None:
        print(f'  |Valid size:\t{len(valid)}')
    print(f'  |Test size:\t{len(test)}')

    return train_loader, valid_loader, test_loader, classes, input_channel

def get_text_dataloaders(dataset_name, valid_size, batch_size, subtrain_ratio=1.0, dataroot='.data'):

    TEXT = data.Field(lower=True, include_lengths=True, batch_first=False)
    LABEL = data.Field(sequential=False)
    fields = {'text': TEXT, 'label': LABEL}
    print(fields)

    if dataset_name == 'sst2':
        train, valid, test = txtdatasets.SST.splits(TEXT, LABEL, root=dataroot)
        train, valid, test = binarize(train), binarize(valid), binarize(test)
        if subtrain_ratio < 1.0:
            train, hold_train = train.split(
                split_ratio=subtrain_ratio, stratified=True)
        classes = ['negative', 'positive']
    elif dataset_name == 'trec':
        random.seed(0)
        train, test = txtdatasets.TREC.splits(
            TEXT, LABEL, fine_grained=False, root=dataroot)
        if valid_size > 0:
            train, valid = train.split(
                stratified=True, random_state=random.getstate())  # default 0.7
        else:
            valid = None
        if subtrain_ratio < 1.0:
            train, hold_train = train.split(
                split_ratio=subtrain_ratio, stratified=True, random_state=random.getstate())
        classes = ['DESC', 'ENTY', 'ABBR', 'HUM', 'NUM', 'LOC']
    else:
        ValueError(f'Invalid dataset name={dataset_name}')

    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300, cache=EMB_DIR))
    LABEL.build_vocab(train)

    train_loader, valid_loader, test_loader = data.BucketIterator.splits(
        (train, valid, test), batch_size=batch_size, sort=True, sort_key=lambda x: len(x.text),
        sort_within_batch=True)

    print('### Dataset ###')
    print(f'=>{dataset_name}')
    print(f'  |Train size:\t{len(train)}')
    if valid is not None:
        print(f'  |Valid size:\t{len(valid)}')
    print(f'  |Test size:\t{len(test)}')
    print(f'  |Vocab size:\t{len(TEXT.vocab)}')

    return train_loader, valid_loader, test_loader, classes, TEXT.vocab

def get_ts_dataloaders(dataset_name, valid_size, batch_size,test_size = 0.2, subtrain_ratio=1.0, dataroot='.data', 
        multilabel=False, default_split=False,labelgroup='',randaug_dic={},fix_policy_list=[],class_wise=False,
        beat_aug=False,info_region=None, rd_seed=None, test_augment=False, fold_assign=[],augselect=None,num_workers=0):
    kwargs = {}
    down_sample = False
    #choose dataset
    if dataset_name == 'ptbxl':
        dataset_func = PTBXL
        if labelgroup:
            kwargs['labelgroup']=labelgroup
    elif dataset_name == 'wisdm':
        dataset_func = WISDM
    elif dataset_name == 'edfx':
        dataset_func = EDFX
        kwargs['max_folds'] = 10
    elif dataset_name == 'chapman':
        dataset_func = Chapman
        if labelgroup:
            kwargs['labelgroup']=labelgroup
    elif dataset_name == 'chapmands':
        down_sample = True
        dataset_func = Chapman
        if labelgroup:
            kwargs['labelgroup']=labelgroup
    elif dataset_name == 'icbeb':
        dataset_func = ICBEB
        if labelgroup:
            kwargs['labelgroup']=labelgroup
    elif dataset_name == 'georgia':
        dataset_func = Georgia
        if labelgroup:
            kwargs['labelgroup']=labelgroup
    else:
        ValueError(f'Invalid dataset name={dataset_name}')
    if down_sample: #for down sample
        dataset_Hz = 100
    else:
        dataset_Hz = dataset_func.Hz
    #rand augment !!! have bug when not using default split
    aug_set = None
    if augselect=='ecg_noise':
        aug_set = ECG_NOISE_DICT
    print('Aug set: ',aug_set)
    train_transfrom = []
    class_wise_transfrom = []
    if randaug_dic.get('randaug',False):
        print('Using RandAugment')
        train_transfrom.extend([
            ToTensor(),
            RandAugment(randaug_dic['rand_n'],randaug_dic['rand_m'],rd_seed=rd_seed,augselect=randaug_dic['augselect'],sfreq=dataset_Hz)])
    if len(fix_policy_list)>0:
        print('Using Transfrom')
        if class_wise:
            print('Class-Wise') #tmp fix of num_class!!!
            class_wise_transfrom.extend([
                ToTensor(),
                TransfromAugment_classwise(fix_policy_list,m=randaug_dic['rand_m'],n=randaug_dic['rand_n'],
                num_class=5,rd_seed=rd_seed,p=randaug_dic['aug_p'],sfreq=dataset_Hz) #tmp!!!
            ])
        elif beat_aug and info_region!=None:
            print('Heart Beat Infomation Region')
            train_transfrom.extend([
                ToTensor(),
                BeatAugment(fix_policy_list,m=randaug_dic['rand_m'],n=randaug_dic['rand_n'],
                mode=info_region,rd_seed=rd_seed,p=randaug_dic['aug_p'],sfreq=dataset_Hz)
            ])
        elif info_region!=None:
            print('Infomation Region')
            train_transfrom.extend([
                ToTensor(),
                InfoRAugment(fix_policy_list,m=randaug_dic['rand_m'],n=randaug_dic['rand_n'],
                mode=info_region,rd_seed=rd_seed,p=randaug_dic['aug_p'],sfreq=dataset_Hz)
            ])
        else:
            train_transfrom.extend([
                ToTensor(),
                TransfromAugment(fix_policy_list,m=randaug_dic['rand_m'],n=randaug_dic['rand_n'],
                rd_seed=rd_seed,p=randaug_dic['aug_p'],sfreq=dataset_Hz,aug_dict=aug_set)
            ])
    print('Train transfrom: ',train_transfrom)
    if test_augment:
        print('Using valid/test transfrom, just for experiment')
        valid_transfrom = train_transfrom
        test_transfrom = train_transfrom
    else:
        valid_transfrom = []
        test_transfrom = []
    
    #split
    if (not default_split or dataset_name=='chapman') and len(fold_assign)==0: #chapman didn't have default split now!!!
        #!!!bug when use transfrom
        dataset = dataset_func(dataroot,multilabel=multilabel,**kwargs)
        total = len(dataset)
        random.seed(0) #!!!
        rd_idxs = [i for i in range(total)]
        random.shuffle(rd_idxs)
        test = Subset(dataset,rd_idxs[:int(total*test_size)])
        if valid_size > 0:
            valid = Subset(dataset,rd_idxs[int(total*test_size):int(total*(test_size+valid_size/10))])
            train_idx = rd_idxs[int(total*(test_size+valid_size/10)):]
            train_idx = train_idx[:int(len(train_idx)*subtrain_ratio)]
            dataset_tr = dataset_func(dataroot,multilabel=multilabel,augmentations=train_transfrom,**kwargs) #!tmp fix
            train = Subset(dataset_tr,train_idx)
    elif len(fold_assign)==3: #train,valid,test
        #dataset raw
        train = dataset_func(dataroot,mode=fold_assign[0],multilabel=multilabel,augmentations=train_transfrom,**kwargs)
        valid = dataset_func(dataroot,mode=fold_assign[1],multilabel=multilabel,augmentations=valid_transfrom,**kwargs)
        test = dataset_func(dataroot,mode=fold_assign[2],multilabel=multilabel,augmentations=test_transfrom,**kwargs)
        #down sample to 100 Hz if needed
        if down_sample:
            print(f'Before down sample {train.input_data.shape}')
            train.down_sample(100)
            valid.down_sample(100)
            test.down_sample(100)
            print(f'After down sample {train.input_data.shape}')
        #preprocess !!!
        ss = StandardScaler()
        print(f'Before dataset {train.input_data.shape} sample 0:')
        print(train[0][0])
        ss = train.fit_preprocess(ss)
        ss = train.trans_preprocess(ss)
        ss = valid.trans_preprocess(ss)
        ss = test.trans_preprocess(ss)
        print(f'After dataset {train.input_data.shape} sample 0:')
        print(train[0][0])
        dataset = train
    else:
        if dataset_name == 'edfx': #edfx have special split method
            dataset = dataset_func(dataroot,multilabel=multilabel,**kwargs)
            splits_proportions = dataset.CV_split_indices() #(k, ratio, sub_tr_idx, valid_idx, test_idx) * 5
            split_info = splits_proportions[0]
            sub_tr_idx, valid_idx, test_idx = split_info[2],split_info[3],split_info[4]
            test = Subset(dataset,test_idx)
            valid = Subset(dataset,valid_idx)
            dataset_tr = dataset_func(dataroot,multilabel=multilabel,augmentations=train_transfrom,**kwargs) #!tmp fix
            train = Subset(dataset_tr,sub_tr_idx)
        else:
            train = dataset_func(dataroot,mode='train',multilabel=multilabel,augmentations=train_transfrom,**kwargs)
            valid = dataset_func(dataroot,mode='valid',multilabel=multilabel,augmentations=valid_transfrom,**kwargs)
            test = dataset_func(dataroot,mode='test',multilabel=multilabel,augmentations=test_transfrom,**kwargs)
            dataset = train
    classes = [i for i in range(dataset.num_class)]
    input_channel = dataset.channel

    train_loader = DataLoader(train,batch_size=batch_size, shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=True)#8/19
    valid_loader = DataLoader(valid,batch_size=batch_size, shuffle=False,num_workers=num_workers,pin_memory=True)#11/9 fix
    test_loader = DataLoader(test,batch_size=batch_size, shuffle=False,num_workers=num_workers,pin_memory=True)#11/9 #fix

    print('### Dataset ###')
    print(f'=>{dataset_name}')
    print(f'  |Train size:\t{len(train)}')
    if valid is not None:
        print(f'  |Valid size:\t{len(valid)}')
    print(f'  |Test size:\t{len(test)}')

    return train_loader, valid_loader, test_loader, classes, input_channel