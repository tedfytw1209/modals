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
from modals.datasets import PTBXL,WISDM,Chapman,EDFX
from modals.operation_tseries import ToTensor,RandAugment,TransfromAugment

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
        multilabel=False, default_split=False,labelgroup='',randaug_dic={},fix_policy_list=[],rd_seed=None):
    kwargs = {}
    #choose dataset
    if dataset_name == 'ptbxl':
        dataset_func = PTBXL
        if labelgroup:
            kwargs['labelgroup']=labelgroup
    elif dataset_name == 'wisdm':
        dataset_func = WISDM
    elif dataset_name == 'edfx':
        dataset_func = EDFX
    elif dataset_name == 'chapman':
        dataset_func = Chapman
    else:
        ValueError(f'Invalid dataset name={dataset_name}')
    #rand augment !!! have bug when not using default split
    train_transfrom = []
    if randaug_dic.get('randaug',False):
        print('Using RandAugment')
        train_transfrom.extend([
            ToTensor(),
            RandAugment(randaug_dic['rand_n'],randaug_dic['rand_m'],rd_seed=rd_seed)])
    if len(fix_policy_list)>0:
        
        train_transfrom.extend([
            ToTensor(),
            TransfromAugment(fix_policy_list,randaug_dic['rand_m'],n=randaug_dic['rand_n'],rd_seed=rd_seed)
            ])
    
    #split
    if not default_split or dataset_name=='chapman': #chapman didn't have default split now!!!
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
            train = Subset(dataset,train_idx)
    else:
        if dataset_name == 'edfx': #edfx have special split method
            dataset = dataset_func(dataroot,multilabel=multilabel,**kwargs)
            splits_proportions = dataset.CV_split_indices() #(k, ratio, sub_tr_idx, valid_idx, test_idx) * 5
            split_info = splits_proportions[0]
            sub_tr_idx, valid_idx, test_idx = split_info[2],split_info[3],split_info[4]
            test = Subset(dataset,test_idx)
            valid = Subset(dataset,valid_idx)
            train = Subset(dataset,sub_tr_idx)
        else:
            train = dataset_func(dataroot,mode='train',multilabel=multilabel,augmentations=train_transfrom,**kwargs)
            valid = dataset_func(dataroot,mode='valid',multilabel=multilabel,**kwargs)
            test = dataset_func(dataroot,mode='test',multilabel=multilabel,**kwargs)
            dataset = train
    classes = [i for i in range(dataset.num_class)]
    input_channel = dataset.channel
    print('Print sample 0')
    samples = train[0] # data,len,label
    print(samples[0])
    print(samples[0].shape)
    print(samples[1])
    print(samples[2])

    train_loader = DataLoader(train,batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)
    valid_loader = DataLoader(valid,batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)
    test_loader = DataLoader(test,batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)

    print('### Dataset ###')
    print(f'=>{dataset_name}')
    print(f'  |Train size:\t{len(train)}')
    if valid is not None:
        print(f'  |Valid size:\t{len(valid)}')
    print(f'  |Test size:\t{len(test)}')

    return train_loader, valid_loader, test_loader, classes, input_channel