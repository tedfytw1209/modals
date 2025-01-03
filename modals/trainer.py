import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import  models
from networks.blstm import BiLSTM,LSTM
from networks.LSTM import LSTM_ecg,LSTM_ptb
from networks.LSTM_attention import LSTM_attention
from networks.MF_transformer import MF_Transformer
from networks.Sleep_stager import SleepStagerChambon2018
from networks.resnet1d import resnet1d_wang,resnet1d101
from networks.xresnet1d import xresnet1d101
from networks.basic_conv1d import make_fcn_wang
from networks.inception1d import make_inception1d
from networks.resnet import ResNet
from networks.wideresnet import WideResNet
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from utility import mixup_criterion, mixup_data, mAP_cw, AUROC_cw

from modals.data_util import get_text_dataloaders,get_ts_dataloaders,get_image_dataloaders
from modals.policy import PolicyManager, RawPolicy

if torch.cuda.is_available():
    import modals.augmentation_transforms as aug_trans
else:
    import modals.augmentation_transforms_cpu as aug_trans

from modals.custom_ops import (HardestNegativeTripletSelector,
                               RandomNegativeTripletSelector,
                               SemihardNegativeTripletSelector)
from modals.losses import (OnlineTripletLoss, adverserial_loss,
                           discriminator_loss)
from modals.operation_tseries import ToTensor,TransfromAugment,InfoRAugment,BeatAugment,TS_OPS_NAMES,TS_ADD_NAMES,ECG_OPS_NAMES
import wandb
import ray.tune as tune

def count_parameters(model):
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f' |Trainable parameters: {temp}')


def build_model(model_name, vocab, n_class, z_size=2, dataset='',max_len=1000,hz=500):
    net = None
    if model_name == 'blstm':
        n_hidden = 256
        config = {'n_vocab': len(vocab),
                  'n_embed': 300,
                  'emb': vocab.vectors,
                  'n_hidden': n_hidden,
                  'n_output': n_class,
                  'n_layers': 2,
                  'pad_idx': vocab.stoi['<pad>'],
                  'b_dir': True,
                  'rnn_drop': 0.2,
                  'fc_drop': 0.5}
        net = BiLSTM(config)
        z_size = n_hidden
    elif model_name == 'lstm':
        n_hidden = 128
        config = {
                  'n_embed': vocab, #input channel in time-series
                  'n_hidden': n_hidden,
                  'n_output': n_class,
                  'n_layers': 1,
                  'b_dir': False,
                  'rnn_drop': 0.2,
                  'fc_drop': 0.5}
        net = LSTM(config)
        z_size = n_hidden
    elif model_name == 'lstm_ecg':
        n_hidden = 512
        config = {
                  'n_embed': vocab,
                  'n_hidden': n_hidden,
                  'n_output': n_class,
                  'n_layers': 2,
                  'b_dir': False,
                  'rnn_drop': 0.2,
                  'fc_drop': 0.5}
        net = LSTM_ecg(config)
        z_size = n_hidden
    elif model_name == 'lstm_ptb':
        n_hidden = 128
        config = {
                  'n_embed': vocab,
                  'n_hidden': n_hidden,
                  'n_output': n_class,
                  'n_layers': 2,
                  'b_dir': False,
                  'concat_pool': True,
                  'rnn_drop': 0.25,
                  'fc_drop': 0.5}
        net = LSTM_ptb(config)
        z_size = n_hidden
    elif model_name == 'mf_trans':
        n_hidden = 256
        config = {'n_output':n_class,'n_embed':vocab,'rnn_drop': 0.2,'fc_drop': 0.5,'max_len':max_len,'hz':hz}
        add_model_config = {}
        add_model_config['seg_config'] = {'seg_ways':'fix', 'rr_method':'pan'}
        mode_config = {
                  'n_hidden': n_hidden,
                  'n_layers': 5,
                  'n_head': 8, #tmp params
                  'n_dff': n_hidden*2, #tmp params
                  'b_dir': False,
                  'concat_pool': True,
                  'rnn_drop': 0.1,
                  'fc_drop': 0.5}
        config.update(mode_config)
        config.update(add_model_config)
        net = MF_Transformer(config)
        z_size = n_hidden * 2 #concat adaptive pool
    elif model_name == 'lstm_atten':
        n_hidden = 512
        config = {
                  'n_embed': vocab,
                  'n_hidden': n_hidden,
                  'n_output': n_class,
                  'n_layers': 1,
                  'b_dir': True,
                  'rnn_drop': 0.2,
                  'fc_drop': 0.5}
        net = LSTM_attention(config)
        z_size = n_hidden
    elif model_name == 'resnet_wang':
        n_hidden = 128
        config = {
                  'input_channels': vocab,
                  'inplanes': n_hidden,
                  'num_classes': n_class,
                  'kernel_size': 5,
                  'lin_ftrs_head': [n_hidden], #8/17 add
                  'ps_head': 0.5}
        net = resnet1d_wang(config)
        z_size = n_hidden * 2 #concat adaptive pool
    elif model_name == 'resnet101':
        n_hidden = 128
        config = {
                  'input_channels': vocab,
                  'inplanes': n_hidden,
                  'num_classes': n_class,
                  'kernel_size': 5,
                  'lin_ftrs_head': [n_hidden], #8/17 add
                  'ps_head': 0.5}
        model_config = {}
        net = resnet1d101(config)
        z_size = n_hidden * 2 #concat adaptive pool
    elif model_name == 'xresnet101':
        #conf_fastai_xresnet1d101 = {'modelname':'fastai_xresnet1d101', 'modeltype':'fastai_model', 'parameters':dict()}
        #elif(self.name.startswith("fastai_xresnet1d101")):
        #    model = xresnet1d101(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        n_hidden = 128
        config = {
                  'input_channels': vocab,
                  #'inplanes': n_hidden,
                  'num_classes': n_class,
                  'kernel_size': 5,
                  'lin_ftrs_head': [n_hidden], #8/17 add
                  'ps_head': 0.5}
        model_config = {}
        net = xresnet1d101(config)
        z_size = n_hidden * 2 #concat adaptive pool
    elif model_name == 'inception':
        n_hidden = 128
        config = {
                  'input_channels': vocab,
                  #'inplanes': n_hidden,
                  'num_classes': n_class,
                  'kernel_size': 5 * 8, # 8 * self.kernel size 
                  'lin_ftrs_head': [n_hidden], #8/17 add
                  'ps_head': 0.5}
        model_config = {}
        net = make_inception1d(config)
        z_size = n_hidden * 2 #concat adaptive pool
    elif model_name == 'fcn_wang':
        n_hidden = 128
        config = {
                  'input_channels': vocab,
                  #'inplanes': n_hidden,
                  'num_classes': n_class,
                  'lin_ftrs_head': [n_hidden], #8/17 add
                  'ps_head': 0.5}
        model_config = {}
        net = make_fcn_wang(config)
        z_size = n_hidden * 2 #concat adaptive pool
    elif model_name == 'cnn_sleep': #
        #n_hidden = 512
        config = {
                  'n_channels': vocab,
                  'dataset': dataset,
                  'batch_norm': True,
                  'n_output': n_class,
                  'fc_drop': 0.25,
                  }
        net = SleepStagerChambon2018(config)
        z_size = net.len_last_layer
    elif model_name == 'densenet131':
        net = models.densenet121(pretrained=True)
        z_size = net.classifier.in_features
        net.classifier = nn.Sequential(nn.Linear(z_size, n_class), nn.Sigmoid()) #change classifer
    elif model_name == 'resnet50': #decoupling_method == 'cRT' is best
        net = models.resnet50(pretrained=True)
        z_size = net.fc.in_features
        net.fc = nn.Sequential(nn.Linear(z_size, n_class), nn.Sigmoid()) #change classifer
    else:
        ValueError(f'Invalid model name={model_name}')

    print('\n### Model ###')
    print(f'=> {model_name}')
    print(f'embedding=> {z_size}')
    count_parameters(net)
    print(net)

    return net, z_size, model_name

def build_img_model(model_name, vocab, n_class, z_size=2):
    net = None
    if model_name == 'resnet50':
        net = ResNet(dataset='imagenet', n_channel=vocab, depth=50, num_classes=n_class, bottleneck=True)
    elif model_name == 'resnet200':
        net = ResNet(dataset='imagenet', n_channel=vocab, depth=200, num_classes=n_class, bottleneck=True)
    elif model_name == 'wresnet40_2':
        net = WideResNet(40, 2, dropout_rate=0.0, num_classes=n_class)
    elif model_name == 'wresnet28_10':
        net = WideResNet(28, 10, dropout_rate=0.0, num_classes=n_class)
    else:
        raise NameError('no model named, %s' % model_name)
    z_size = net.fc.in_features

    print('\n### Model ###')
    print(f'=> {model_name}')
    print(f'embedding=> {z_size}')
    count_parameters(net)
    print(net)

    return net, z_size, model_name

def reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)

class Discriminator(nn.Module):
    def __init__(self, z_size):
        super(Discriminator, self).__init__()
        self.z_size = z_size
        self.fc1 = nn.Linear(z_size, 256) #!follow paper
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class TextModelTrainer(object):

    def __init__(self, hparams, name=''):
        self.hparams = hparams
        print(hparams)
        self.multilabel = False
        self.name = name

        random.seed(0)
        self.train_loader, self.valid_loader, self.test_loader, self.classes, self.vocab = get_text_dataloaders(
            hparams['dataset_name'], valid_size=hparams['valid_size'], batch_size=hparams['batch_size'],
            subtrain_ratio=hparams['subtrain_ratio'], dataroot=hparams['dataset_dir'])
        random.seed()

        self.device = torch.device(
            hparams['gpu_device'] if torch.cuda.is_available() else 'cpu')
        print()
        print('### Device ###')
        print(self.device)
        self.net, self.z_size, self.file_name = build_model(
            hparams['model_name'], self.vocab, len(self.classes),dataset=hparams['dataset_name'])
        self.net = self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        if hparams['mode'] in ['train', 'search']:
            self.optimizer = optim.Adam(self.net.parameters(), self.hparams['lr']) #follow paper
            self.loss_dict = {'train': [], 'valid': []}

            if hparams['use_modals']:
                print("\n=> ### Policy ###")
                # print(f'  |hp_policy: {hparams['hp_policy']}')
                # print(f'  |policy_path: {hparams['policy_path']}')
                raw_policy = RawPolicy(mode=hparams['mode'], num_epochs=hparams['num_epochs'],
                                       hp_policy=hparams['hp_policy'], policy_path=hparams['policy_path'])
                transformations = aug_trans
                self.pm = PolicyManager(
                    transformations, raw_policy, len(self.classes), self.device)

            print("\n### Loss ###")
            print('Classification Loss')

            if hparams['mixup']:
                print('Mixup')

            if hparams['enforce_prior']:
                print('Adversarial Loss')
                self.EPS = 1e-15
                self.D = Discriminator(self.z_size)
                self.D = self.D.to(self.device)
                self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.01, #!follow paper
                                              weight_decay=hparams['wd'])
                # self.G_optimizer = optim.Adam(self.net.parameters(), lr=0.001)

            if hparams['metric_learning']:
                margin = hparams['metric_margin']
                metric_loss = hparams["metric_loss"]
                metric_weight = hparams["metric_weight"]
                print(
                    f"Metric Loss (margin: {margin} loss: {metric_loss} weight: {metric_weight})")

                self.M_optimizer = optim.SGD(
                    self.net.parameters(), momentum=0.9, lr=1e-3, weight_decay=1e-8)
                self.metric_weight = hparams['metric_weight']

                if metric_loss == 'random':
                    self.metric_loss = OnlineTripletLoss(
                        margin, RandomNegativeTripletSelector(margin))
                elif metric_loss == 'hardest':
                    self.metric_loss = OnlineTripletLoss(
                        margin, HardestNegativeTripletSelector(margin))
                elif metric_loss == 'semihard':
                    self.metric_loss = OnlineTripletLoss(
                        margin, SemihardNegativeTripletSelector(margin))

    def reset_model(self, z_size=256):
        # tunable z_size only use for visualization
        # if blstm is used, it is automatically 256
        self.net, self.z_size, self.file_name = build_model(
            self.hparams['model_name'], self.vocab, len(self.classes), z_size)
        self.net = self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), self.hparams['lr'])
        self.loss_dict = {'train': [], 'valid': []}

    def reset_discriminator(self, z_size=256):
        self.D = Discriminator(z_size)
        self.D = self.D.to(self.device)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.01, #!follow paper
                                     weight_decay=self.hparams['wd'])

    def update_policy(self, policy):
        raw_policy = RawPolicy(mode='train', num_epochs=1,
                               hp_policy=policy, policy_path=None)
        self.pm.update_policy(raw_policy)

    def _train(self, cur_epoch):
        self.net.train()
        self.net.training = True
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, len(self.train_loader))  # cosine learning rate
        
        train_losses = 0.0
        clf_losses = 0.0
        metric_losses = 0.0
        d_losses = 0.0
        g_losses = 0.0
        correct = 0
        total = 0
        n_batch = len(self.train_loader)

        print(f'\n=> Training Epoch #{cur_epoch}')
        for batch_idx, batch in enumerate(self.train_loader):

            inputs, seq_lens, labels = batch.text[0].to(
                self.device), batch.text[1].to(self.device), batch.label.to(self.device)

            # if self.hparams['dataset_name'] == 'sst2':
            labels -= 1  # because I binarized the data
            seed_features = self.net.extract_features(inputs, seq_lens)
            features = seed_features

            if self.hparams['manifold_mixup']:
                features, targets_a, targets_b, lam = mixup_data(features, labels,
                                                                 0.2, use_cuda=True)
                features, targets_a, targets_b = map(Variable, (features,
                                                                targets_a, targets_b))
            # apply pba transformation
            if self.hparams['use_modals']:
                features = self.pm.apply_policy(
                    features, labels, cur_epoch, batch_idx, verbose=1).to(self.device)

            outputs = self.net.classify(features)  # Forward Propagation

            if self.hparams['mixup']:
                inputs, targets_a, targets_b, lam = mixup_data(outputs, labels,
                                                               self.hparams['alpha'], use_cuda=True)
                inputs, targets_a, targets_b = map(Variable, (outputs,
                                                              targets_a, targets_b))
            # freeze D
            if self.hparams['enforce_prior']:
                for p in self.D.parameters():
                    p.requires_grad = False

            # classification loss
            if self.hparams['mixup'] or self.hparams['manifold_mixup']:
                c_loss = mixup_criterion(
                    self.criterion, outputs, targets_a, targets_b, lam)
            else:
                c_loss = self.criterion(outputs, labels)  # Loss
            clf_losses += c_loss.item()

            # total loss
            loss = c_loss
            if self.hparams['metric_learning']:
                m_loss = self.metric_loss(seed_features, labels)[0]
                metric_losses += m_loss.item()
                loss = self.metric_weight * m_loss + \
                    (1-self.metric_weight) * c_loss

            train_losses += loss.item()

            if self.hparams['enforce_prior']:
                # Regularizer update
                # freeze D
                for p in self.D.parameters():
                    p.requires_grad = False
                self.net.train()
                d_fake = self.D(features)
                g_loss = self.hparams['prior_weight'] * \
                    adverserial_loss(d_fake, self.EPS)
                g_losses += g_loss.item()
                loss += g_loss

            self.optimizer.zero_grad()
            loss.backward()  # Backward Propagation
            clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()  # Optimizer update

            if self.hparams['enforce_prior']:
                # Discriminator update
                for p in self.D.parameters():
                    p.requires_grad = True

                features = self.net.extract_features(inputs, seq_lens)
                d_real = self.D(torch.randn(features.size()).to(self.device))
                d_fake = self.D(F.softmax(features, dim=0))
                d_loss = discriminator_loss(d_real, d_fake, self.EPS)
                self.D_optimizer.zero_grad()
                d_loss.backward()
                self.D_optimizer.step()
                d_losses += d_loss.item()

            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if self.hparams['mixup']: #how this acc means?
                correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                            + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            else:
                correct += (predicted == labels).sum().item()

        # step
        step = (cur_epoch-1)*(len(self.train_loader)) + batch_idx
        total_steps = self.hparams['num_epochs']*len(self.train_loader)

        # logs
        display = f'| Epoch [{cur_epoch}/{self.hparams["num_epochs"]}]\tIter[{step}/{total_steps}]\tLoss: {train_losses/n_batch:.4f}\tAcc@1: {correct/total:.4f}\tclf_loss: {clf_losses/n_batch:.4f}'
        if self.hparams['enforce_prior']:
            display += f'\td_loss: {d_losses/n_batch:.4f}\tg_loss: {g_losses/n_batch:.4f}'
        if self.hparams['metric_learning']:
            display += f'\tmetric_loss: {metric_losses/n_batch:.4f}'
        print(display)

        return correct/total, train_losses/total

    def _test(self, cur_epoch, mode):
        self.net.eval()
        self.net.training = False
        correct = 0
        total = 0
        test_loss = 0.0
        data_loader = self.valid_loader if mode == 'valid' else self.test_loader
        confusion_matrix = torch.zeros(len(self.classes), len(self.classes))
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs, seq_lens, labels = batch.text[0].to(
                    self.device), batch.text[1].to(self.device), batch.label.to(self.device)

                # if self.hparams['dataset_name'] == 'sst2':
                labels -= 1  # because I binarized the data

                outputs = self.net(inputs, seq_lens)
                loss = self.criterion(outputs, labels)  # Loss
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                torch.cuda.empty_cache()

            print(
                f'| ({mode}) Epoch #{cur_epoch}\t Loss: {test_loss/total:.4f}\t Acc@1: {correct/total:.4f}')
            #class-wise
            cw_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
            print(f'class-wise Acc: ',cw_acc)

        return correct/total, test_loss/total

    def run_model(self, epoch):
        if self.hparams['use_modals']:
            self.pm.reset_text_data_pool(
                self.net, self.train_loader, self.hparams['temperature'], self.hparams['distance_metric'], self.hparams['dataset_name'])

        train_acc, tl = self._train(epoch)
        self.loss_dict['train'].append(tl)

        if self.hparams['valid_size'] > 0:
            val_acc, vl = self._test(epoch, mode='valid')
            self.loss_dict['valid'].append(vl)
        else:
            val_acc = 0.0

        return train_acc, val_acc

    # for benchmark
    def save_checkpoint(self, ckpt_dir, epoch, title=''):
        if self.hparams.get('base_path',''):
            ckpt_dir = os.path.join(self.hparams.get('base_path',''),ckpt_dir)
        if self.hparams.get('kfold',-1)>=0:
            test_fold_idx = self.hparams['kfold']
            add_word = f'_fold{test_fold_idx}'
        else:
            add_word = ''
        path = os.path.join(
            ckpt_dir, self.hparams['dataset_name'], f'{self.name}{add_word}_{self.file_name}{title}')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        torch.save({'state': self.net.state_dict(),
                    'epoch': epoch,
                    'loss': self.loss_dict,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}, path)

        print(f'=> saved the model {self.file_name} to {path}')
        return path

    # for ray
    def save_model(self, ckpt_dir, epoch):
        # save the checkpoint.
        print(self.file_name)
        print(ckpt_dir)
        path = os.path.join(ckpt_dir, self.file_name)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        torch.save({'state': self.net.state_dict(),
                    'epoch': epoch,
                    'loss': self.loss_dict,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}, path)

        print(f'=> saved the model {self.file_name} to {path}')
        return path

    def load_model(self, ckpt):
        # load the checkpoint.
        # path = os.path.join(ckpt_dir, self.model_name)
        # map_location='cuda:0')
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        print('Model keys: ',[n for n in checkpoint.keys()])
        if '.pth' in ckpt:
            self.net.load_state_dict(checkpoint['model'])
        else:
            self.net.load_state_dict(checkpoint['state'])
        self.loss_dict = checkpoint['loss']
        if self.hparams['mode'] != 'test':
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        print(f'=> loaded checkpoint of {self.file_name} from {ckpt}')
        return checkpoint['epoch'], checkpoint['loss']

    def reset_config(self, new_hparams):
        self.hparams = new_hparams
        new_policy = RawPolicy(mode=self.hparams['mode'], num_epochs=self.hparams['num_epochs'],
                               hp_policy=self.hparams['hp_policy'])
        self.pm.update_policy(new_policy)
        return

class TSeriesModelTrainer(TextModelTrainer):
    def __init__(self, hparams, name=''):
        self.hparams = hparams
        print(hparams)
        #random seed setting
        reproducibility(hparams['seed'])
        #wandb.config.update(hparams)
        if name:
            self.name = name
        else:
            self.name = hparams.get('name','')
        self.multilabel = hparams['multilabel']
        self.randaug_dic = {'randaug':hparams.get('randaug',False),'rand_n':hparams.get('rand_n',0),
            'rand_m':hparams.get('rand_m',0),'augselect':hparams.get('augselect',''),'aug_p':hparams.get('aug_p',0.5)}
        print('Rand Augment: ',self.randaug_dic)
        fix_policy = hparams['fix_policy']
        if fix_policy==None:
            fix_policy = []
        elif fix_policy=='ts_base':
            fix_policy = TS_OPS_NAMES
        elif fix_policy=='ts_add':
            fix_policy = TS_OPS_NAMES + TS_ADD_NAMES
        elif ',' in fix_policy:
            fix_policy = fix_policy.split(',')
        else:
            fix_policy = [fix_policy]
        self.fix_policy = fix_policy
        self.info_region = hparams.get('info_region',None)
        #print('Trainer get info region:',self.info_region)
        self.beat_aug = hparams.get('beat_aug',False)
        #kfold or not
        train_val_test_folds = []
        if hparams['kfold']==10:
            test_fold_idx = tune.suggest.repeater.TRIAL_INDEX
        elif hparams['kfold']>=0:
            test_fold_idx = hparams['kfold']
        else:
            test_fold_idx = -1
        if test_fold_idx>=0:
            train_val_test_folds = [[],[],[]] #train,valid,test
            for i in range(10):
                curr_fold = (i+test_fold_idx)%10 +1 #fold is 1~10
                if i==0:
                    train_val_test_folds[2].append(curr_fold)
                elif i==9:
                    train_val_test_folds[1].append(curr_fold)
                else:
                    train_val_test_folds[0].append(curr_fold)
            print('Train/Valid/Test fold split ',train_val_test_folds)
        self.train_loader, self.valid_loader, self.test_loader, self.classes, self.vocab = get_ts_dataloaders(
            hparams['dataset_name'], valid_size=hparams['valid_size'], batch_size=hparams['batch_size'],
            subtrain_ratio=hparams['subtrain_ratio'], dataroot=hparams['dataset_dir'],multilabel=self.multilabel,
            default_split=hparams['default_split'],labelgroup=hparams['labelgroup'],randaug_dic=self.randaug_dic,
            fix_policy_list=fix_policy,class_wise=hparams['class_wise'],info_region=self.info_region, beat_aug=self.beat_aug,
            fold_assign=train_val_test_folds ,augselect=hparams['augselect'],num_workers=hparams['num_workers'],rd_seed=hparams['seed']
            )
        self.device = torch.device(
            hparams['gpu_device'] if torch.cuda.is_available() else 'cpu')
        #model
        print()
        print('### Device ###')
        print(self.device)
        #tmp
        if hparams['dataset_name']=='chapman':
            self.hz = 500
            self.max_len = 5000
        elif hparams['dataset_name']=='icbeb':
            self.hz = 100
            self.max_len = 6000
        else:
            self.hz = 100
            self.max_len = 1000

        self.net, self.z_size, self.file_name = build_model(hparams['model_name'], self.vocab, len(self.classes),max_len=self.max_len,hz=self.hz)
        self.net = self.net.to(self.device)

        if self.multilabel:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.criterion = nn.CrossEntropyLoss()
        if hparams['mode'] in ['train', 'search']:
            self.optimizer = optim.AdamW(self.net.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['wd']) #follow ptbxl batchmark
            self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.hparams['lr'], epochs = self.hparams['num_epochs'], steps_per_epoch = len(self.train_loader))
            #10/29 warmup add
            if not hparams['notwarmup'] and hparams['mode']=='train':
                print('Using warmup scheduler as AdaAug')
                m, e = 2,3
                self.scheduler = GradualWarmupScheduler( #paper not mention!!!
                    self.optimizer,
                    multiplier=m,
                    total_epoch=e,
                    after_scheduler=self.scheduler)
            else:
                print('Not using warmup scheduler')
            self.grad_clip = hparams['gradient_clipping_by_global_norm']
            self.loss_dict = {'train': [], 'valid': []}
            if hparams['use_modals']:
                print("\n=> ### Policy ###")
                raw_policy = RawPolicy(mode=hparams['mode'], num_epochs=hparams['num_epochs'],
                                       hp_policy=hparams['hp_policy'], policy_path=hparams['policy_path'])
                transformations = aug_trans
                self.pm = PolicyManager(
                    transformations, raw_policy, len(self.classes), self.device, multilabel=hparams['multilabel'])
            print("\n### Loss ###")
            print('Classification Loss')
            if hparams['mixup']:
                print('Mixup')
            if hparams['enforce_prior']:
                print('Adversarial Loss')
                self.EPS = 1e-15
                self.D = Discriminator(self.z_size)
                self.D = self.D.to(self.device)
                self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.01, #!follow paper
                                              weight_decay=hparams['wd'])
            if hparams['metric_learning']:
                margin = hparams['metric_margin']
                metric_loss = hparams["metric_loss"]
                metric_weight = hparams["metric_weight"]
                print(
                    f"Metric Loss (margin: {margin} loss: {metric_loss} weight: {metric_weight})")
                self.M_optimizer = optim.SGD(
                    self.net.parameters(), momentum=0.9, lr=1e-3, weight_decay=1e-8)
                self.metric_weight = hparams['metric_weight']
                if metric_loss == 'random':
                    self.metric_loss = OnlineTripletLoss(
                        margin, RandomNegativeTripletSelector(margin))
                elif metric_loss == 'hardest':
                    self.metric_loss = OnlineTripletLoss(
                        margin, HardestNegativeTripletSelector(margin))
                elif metric_loss == 'semihard':
                    self.metric_loss = OnlineTripletLoss(
                        margin, SemihardNegativeTripletSelector(margin))
        # load model
        self.start_epoch = 0
        self.epoch = hparams['num_epochs']
        if hparams.get('restore',None) is not None:
            start_epoch, _ = self.load_model(hparams['restore'])
            self.start_epoch = hparams['num_epochs'] #tmp
    
    def _train(self, cur_epoch, trail_id, training=True):
        if training:
            self.net.train()
            self.net.training = True
        else:
            self.net.eval()
            self.net.training = False
        '''self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, len(self.train_loader))  # cosine learning rate'''
        #follow ptbxl benchmark !!!
        train_losses = 0.0
        clf_losses = 0.0
        metric_losses = 0.0
        d_losses = 0.0
        g_losses = 0.0
        correct = 0
        total = 0
        n_batch = len(self.train_loader)
        confusion_matrix = torch.zeros(len(self.classes), len(self.classes))
        preds = []
        targets = []
        print(f'\n=> Training Epoch #{cur_epoch}')
        for batch_idx, batch in enumerate(self.train_loader):
            inputs, seq_lens, labels = batch[0].float().to(
                self.device), batch[1].cpu(), batch[2].long().to(self.device)
            seed_features = self.net.extract_features(inputs, seq_lens)
            features = seed_features
            if self.hparams['manifold_mixup']:
                features, targets_a, targets_b, lam = mixup_data(features, labels,
                                                                 0.2, use_cuda=True)
                features, targets_a, targets_b = map(Variable, (features,
                                                                targets_a, targets_b))
            # apply pba transformation
            if self.hparams['use_modals']:
                try:
                    features = self.pm.apply_policy(
                        features, labels, cur_epoch, batch_idx, verbose=1).to(self.device)
                except Exception as e:
                    print(e)
                    print('tmp ignore error')
            outputs = self.net.classify(features)  # Forward Propagation
            if self.hparams['mixup']:
                inputs, targets_a, targets_b, lam = mixup_data(outputs, labels,
                                                               self.hparams['alpha'], use_cuda=True)
                inputs, targets_a, targets_b = map(Variable, (outputs,
                                                              targets_a, targets_b))
            # freeze D
            if self.hparams['enforce_prior']:
                for p in self.D.parameters():
                    p.requires_grad = False
            # classification loss
            if self.hparams['mixup'] or self.hparams['manifold_mixup']:
                c_loss = mixup_criterion(
                    self.criterion, outputs, targets_a, targets_b, lam)
            else:
                if self.multilabel:
                    c_loss = self.criterion(outputs, labels.float())  # Loss
                else:
                    c_loss = self.criterion(outputs, labels.long())  # Loss
            clf_losses += c_loss.item()
            # total loss
            loss = c_loss
            if self.hparams['metric_learning']:
                m_loss = self.metric_loss(seed_features, labels)[0]
                metric_losses += m_loss.item()
                loss = self.metric_weight * m_loss + \
                    (1-self.metric_weight) * c_loss
            train_losses += loss.item()

            if self.hparams['enforce_prior']:
                # Regularizer update
                # freeze D
                for p in self.D.parameters():
                    p.requires_grad = False
                self.net.train()
                d_fake = self.D(features)
                g_loss = self.hparams['prior_weight'] * \
                    adverserial_loss(d_fake, self.EPS)
                g_losses += g_loss.item()
                loss += g_loss

            self.optimizer.zero_grad()
            if training:
                loss.backward()  # Backward Propagation
                clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.optimizer.step()  # Optimizer update
                try: #tmp
                    self.scheduler.step()
                except Exception as e:
                    print('Exception:')
                    print(e)

            if self.hparams['enforce_prior']:
                # Discriminator update
                for p in self.D.parameters():
                    p.requires_grad = True

                features = self.net.extract_features(inputs, seq_lens)
                d_real = self.D(torch.randn(features.size()).to(self.device))
                d_fake = self.D(F.softmax(features, dim=0))
                d_loss = discriminator_loss(d_real, d_fake, self.EPS)
                self.D_optimizer.zero_grad()
                d_loss.backward()
                self.D_optimizer.step()
                d_losses += d_loss.item()

            # Accuracy / AUROC
            if not self.multilabel:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if self.hparams['mixup']:
                    correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                            + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
                else:
                    correct += (predicted == labels).sum().item()
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            else:
                predicted = torch.sigmoid(outputs).cpu().detach()
                if torch.any(torch.isnan(predicted)):
                    print('Inputs:', inputs.shape)
                    print(torch.max(inputs))
                    print(torch.min(inputs))
                    print('Representation: ', features.shape)
                    print(features)
                    print(torch.min(features))
                    print('Labels: ', labels.shape)
                    print(labels)
                    print(predicted)
                    print(outputs)
                    assert False
            preds.append(predicted)
            targets.append(labels.cpu().detach())
        
        if not self.multilabel:
            perfrom = 100 * correct/total
            perfrom_cw = 100 * confusion_matrix.diag()/(confusion_matrix.sum(1)+1e-9)
        else:
            targets_np = torch.cat(targets).numpy()
            preds_np = torch.cat(preds).numpy()
            try:
                perfrom = 100 * roc_auc_score(targets_np, preds_np,average='macro')
            except Exception as e:
                print('target shape: ',targets_np.shape)
                print('preds shape: ',preds_np.shape)
                nan_count = np.sum(np.isnan(preds_np))
                inf_count = np.sum(np.isinf(preds_np))
                print('predict nan, inf count: ',nan_count,inf_count)
                print(np.sum(targets_np,axis=0))
                print(np.sum(preds_np,axis=0))
                print(preds_np[np.isnan(preds_np)])
                raise e
            perfrom_cw = AUROC_cw(targets_np,preds_np)
            perfrom = perfrom_cw.mean()
        epoch_loss = train_losses/n_batch
        # step
        step = (cur_epoch-1)*(len(self.train_loader)) + batch_idx
        total_steps = self.hparams['num_epochs']*len(self.train_loader)
        #wandb dic
        out_dic = {}
        out_dic[f'train_loss'] = epoch_loss
        out_dic[f'train_clfloss'] = clf_losses/n_batch
        # logs
        display = f'| Epoch [{cur_epoch}/{self.hparams["num_epochs"]}]\tIter[{step}/{total_steps}]\tLoss: {epoch_loss:.4f}\tAcc@1/MacromAP: {perfrom:.4f}\tclf_loss: {clf_losses/n_batch:.4f}'
        if self.hparams['enforce_prior']:
            display += f'\td_loss: {d_losses/n_batch:.4f}\tg_loss: {g_losses/n_batch:.4f}'
            out_dic[f'train_d_loss'] = d_losses/n_batch
            out_dic[f'train_g_loss'] = g_losses/n_batch
        if self.hparams['metric_learning']:
            display += f'\tmetric_loss: {metric_losses/n_batch:.4f}'
            out_dic[f'train_metric_loss'] = metric_losses/n_batch
        print(display)
        if self.multilabel:
            ptype = 'auroc'
        else:
            ptype = 'acc'
        out_dic[f'train_{ptype}_avg'] = perfrom
        for i,e_c in enumerate(perfrom_cw):
            out_dic[f'train_{ptype}_c{i}'] = e_c

        return perfrom, epoch_loss, out_dic

    def _test(self, cur_epoch, trail_id, mode):
        self.net.eval()
        self.net.training = False
        correct = 0
        total = 0
        test_loss = 0.0
        if mode == 'train':
            data_loader = self.train_loader
        elif mode == 'valid':
            data_loader = self.valid_loader
        else:
            data_loader = self.test_loader
        confusion_matrix = torch.zeros(len(self.classes), len(self.classes))
        preds = []
        targets = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs, seq_lens, labels = batch[0].float().to( #need to follow!!!
                    self.device), batch[1].to(self.device), batch[2].to(self.device)

                outputs = self.net(inputs, seq_lens)
                if self.multilabel:
                    loss = self.criterion(outputs, labels.float())  # Loss multilabel
                else:
                    loss = self.criterion(outputs, labels.long())  # Loss singlelabel
                test_loss += loss.item()

                if not self.multilabel:
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                else:
                    predicted = torch.sigmoid(outputs.data)
                preds.append(predicted.cpu().detach())
                targets.append(labels.cpu().detach().long())
                
                torch.cuda.empty_cache()
        #prediction output
        output_dic = {}
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
        output_dic[f'{mode}_target'] = targets_np
        output_dic[f'{mode}_predict'] = preds_np
        #class-wise
        if not self.multilabel:
            perfrom = 100 * correct/total
            perfrom_cw = 100 * confusion_matrix.diag() / (confusion_matrix.sum(1)+1e-9)
        else:
            perfrom_cw = AUROC_cw(targets_np,preds_np)
            perfrom = perfrom_cw.mean()
        epoch_loss = test_loss / len(data_loader)

        if not self.multilabel:
            print(f'| ({mode}) Epoch #{cur_epoch}\t Loss: {epoch_loss:.4f}\t Acc@1: {perfrom:.4f}')
            print(f'class-wise Acc: ','['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        else:
            print(f'| ({mode}) Epoch #{cur_epoch}\t Loss: {epoch_loss:.4f}\t MacroAUROC: {perfrom:.4f}')
            print(f'class-wise AUROC: ','['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        #wandb dic
        out_dic = {}
        out_dic[f'{mode}_loss'] = epoch_loss
        if self.multilabel:
            ptype = 'auroc'
        else:
            ptype = 'acc'
        out_dic[f'{mode}_{ptype}_avg'] = perfrom
        for i,e_c in enumerate(perfrom_cw):
            out_dic[f'{mode}_{ptype}_c{i}'] = e_c

        return perfrom, epoch_loss, out_dic, output_dic

    def run_model(self, epoch, trail_id):
        if self.hparams['use_modals']:
            try: #tmp fix
                self.pm.reset_tseries_data_pool(
                    self.net, self.train_loader, self.hparams['temperature'], self.hparams['distance_metric'], self.hparams['dataset_name'])
            except Exception as e:
                print(e)
                print('tmp fix')
        #restore setting
        cur_epoch = self.start_epoch + epoch
        if cur_epoch > self.epoch:
            return 0.0, 0.0 , {}, {}
        elif cur_epoch==self.epoch:
            print('Evaluating Train/Valid/Test dataset')
            training = False
        else:
            training = True

        train_acc, tl, train_dic = self._train(epoch, trail_id,training=training)
        self.loss_dict['train'].append(tl)
        if self.hparams['valid_size'] > 0:
            val_acc, vl,val_dic,val_output_dic = self._test(epoch, trail_id, mode='valid')
            self.loss_dict['valid'].append(vl)
            train_dic.update(val_dic)
        else:
            val_acc = 0.0

        return train_acc, val_acc, train_dic, val_output_dic
    #change augment
    def change_augment(self,new_m):
        #not good code !!!
        print('Setting new augment m to ', new_m)
        if self.beat_aug and self.info_region:
            new_augment = [
            ToTensor(),
            BeatAugment(self.fix_policy,m=new_m,n=self.hparams['rand_n'],mode=self.info_region,p=self.hparams['aug_p'])
            ]
        elif self.info_region:
            new_augment = [
            ToTensor(),
            InfoRAugment(self.fix_policy,m=new_m,n=self.hparams['rand_n'],mode=self.info_region,p=self.hparams['aug_p'])
            ]
        else:
            new_augment = [
            ToTensor(),
            TransfromAugment(self.fix_policy,new_m,n=self.hparams['rand_n'],p=self.hparams['aug_p'])
            ]
        self.train_loader.dataset.augmentations = new_augment
        self.valid_loader.dataset.augmentations = new_augment
        self.test_loader.dataset.augmentations = new_augment
    # for benchmark
    def save_checkpoint(self, ckpt_dir, epoch, title='',trail_id=None):
        if self.hparams.get('base_path',''):
            ckpt_dir = os.path.join(self.hparams.get('base_path',''),ckpt_dir)
        add_word = ''
        sub_word = ''
        if self.hparams.get('kfold',-1)>=0:
            test_fold_idx = self.hparams['kfold']
            sub_word += f'fold{test_fold_idx}'
        if self.fix_policy:
            rand_m = self.hparams.get('rand_m',0)
            add_word += f'_{self.fix_policy}{rand_m}'
        if self.hparams['use_modals']:
            trail_word = str(trail_id)
        else:
            trail_word = ''
        dir_path = os.path.join(
            ckpt_dir, self.hparams['dataset_name'], f'{self.name}{add_word}_{self.file_name}',sub_word,trail_word)
        path = os.path.join(dir_path,title)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        torch.save({'state': self.net.state_dict(),
                    'epoch': epoch,
                    'loss': self.loss_dict,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}, path)

        print(f'=> saved the model {self.file_name} to {path}')
        return path
    #for pred
    def save_pred(self, target, pred, ckpt_dir, title='',trail_id=None):
        if self.hparams.get('base_path',''):
            ckpt_dir = os.path.join(self.hparams.get('base_path',''),ckpt_dir)
        add_word = ''
        sub_word = ''
        if self.hparams.get('kfold',-1)>=0:
            test_fold_idx = self.hparams['kfold']
            sub_word += f'fold{test_fold_idx}'
        if self.fix_policy:
            rand_m = self.hparams.get('rand_m',0)
            add_word += f'_{self.fix_policy}{rand_m}'
        if self.hparams['use_modals']:
            trail_word = str(trail_id)
        else:
            trail_word = ''
        dir_path = os.path.join(
            ckpt_dir, self.hparams['dataset_name'], f'{self.name}{add_word}_{self.file_name}',sub_word,trail_word)
        path = os.path.join(dir_path,title)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        col_names = ['target','predict']
        if len(target.shape)>1 and target.shape[1]>1: #multilabel
            col_names = ['target_'+str(i) for i in range(target.shape[1])] + ['predict_'+str(i) for i in range(target.shape[1])]
        else:
            target = target.reshape((-1,1))
            pred = pred.reshape((-1,1))
        out_np = np.concatenate((target,pred),axis=1)
        out_data = pd.DataFrame(out_np,columns=col_names)
        out_data.to_csv(path+'.csv')
        print(f'=> saved the prediction {self.file_name} to {path}')
        return path
    def load_model(self, ckpt,trail_id=None):
        path = ckpt
        print(f'=> loaded checkpoint of {self.file_name} from {path}')
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        print('Model keys: ',[n for n in checkpoint.keys()])
        if '.pth' in path:
            self.net.load_state_dict(checkpoint['model'])
        else:
            self.net.load_state_dict(checkpoint['state'])
        self.loss_dict = checkpoint['loss']
        if self.hparams['mode'] != 'test':
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['epoch'], checkpoint['loss']
    # for ray
    def save_model(self, ckpt_dir, epoch):
        # save the checkpoint.
        print(self.file_name)
        print(ckpt_dir)
        path = os.path.join(ckpt_dir, self.file_name)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        torch.save({'state': self.net.state_dict(),
                    'epoch': epoch,
                    'loss': self.loss_dict,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}, path)

        print(f'=> saved the model {self.file_name} to {path}')
        return path
    
class ImageModelTrainer(TSeriesModelTrainer):
    def __init__(self, hparams, name=''):
        self.hparams = hparams
        print(hparams)
        #random seed setting
        reproducibility(hparams['seed'])
        #wandb.config.update(hparams)
        if name:
            self.name = name
        else:
            self.name = hparams.get('name','')
        self.multilabel = hparams['multilabel']
        self.randaug_dic = {'randaug':hparams.get('randaug',False),'rand_n':hparams.get('rand_n',0),
            'rand_m':hparams.get('rand_m',0),'augselect':hparams.get('augselect',''),'aug_p':hparams.get('aug_p',0.5)}
        print('Rand Augment: ',self.randaug_dic)
        fix_policy = hparams['fix_policy']
        if fix_policy==None:
            fix_policy = []
        elif ',' in fix_policy:
            fix_policy = fix_policy.split(',')
        else:
            fix_policy = [fix_policy]
        self.fix_policy = fix_policy
        #kfold or not
        train_val_test_folds = []
        if hparams['kfold']==10:
            test_fold_idx = -1 #tmp change
        elif hparams['kfold']>=0:
            test_fold_idx = hparams['kfold']
        else:
            test_fold_idx = -1
        if test_fold_idx>=0:
            train_val_test_folds = [[],[],[]] #train,valid,test
            for i in range(10):
                curr_fold = (i+test_fold_idx)%10 +1 #fold is 1~10
                if i==0:
                    train_val_test_folds[2].append(curr_fold)
                elif i==9:
                    train_val_test_folds[1].append(curr_fold)
                else:
                    train_val_test_folds[0].append(curr_fold)
            print('Train/Valid/Test fold split ',train_val_test_folds)
        self.train_loader, self.valid_loader, self.test_loader, self.classes, self.vocab = get_image_dataloaders(
                        hparams['dataset_name'], valid_size=hparams['valid_size'], batch_size=hparams['batch_size'], dataroot=hparams['dataset_dir'],
                        augselect=hparams['augselect'],randaug_dic=self.randaug_dic,fix_policy_list=fix_policy,class_wise=hparams['class_wise'],
                        info_region=None,test_augment=False,num_workers=hparams['num_workers'],rd_seed=hparams['seed'])
        self.device = torch.device(
            hparams['gpu_device'] if torch.cuda.is_available() else 'cpu')
        #model
        print()
        print('### Device ###')
        print(self.device)

        self.net, self.z_size, self.file_name = build_img_model(hparams['model_name'], self.vocab, len(self.classes))
        self.net = self.net.to(self.device)

        if self.multilabel:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.criterion = nn.CrossEntropyLoss()
        if hparams['mode'] in ['train', 'search']:
            self.optimizer = optim.AdamW(self.net.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['wd']) #follow ptbxl batchmark
            self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.hparams['lr'], epochs = self.hparams['num_epochs'], steps_per_epoch = len(self.train_loader))
            # warmup add
            if not hparams['notwarmup'] and hparams['mode']=='train':
                print('Using warmup scheduler as AdaAug')
                m, e = 2,3
                self.scheduler = GradualWarmupScheduler( #paper not mention!!!
                    self.optimizer,
                    multiplier=m,
                    total_epoch=e,
                    after_scheduler=self.scheduler)
            else:
                print('Not using warmup scheduler')
            self.grad_clip = hparams['gradient_clipping_by_global_norm']
            self.loss_dict = {'train': [], 'valid': []}
            if hparams['use_modals']:
                print("\n=> ### Policy ###")
                raw_policy = RawPolicy(mode=hparams['mode'], num_epochs=hparams['num_epochs'],
                                       hp_policy=hparams['hp_policy'], policy_path=hparams['policy_path'])
                transformations = aug_trans
                self.pm = PolicyManager(
                    transformations, raw_policy, len(self.classes), self.device, multilabel=hparams['multilabel'])
            print("\n### Loss ###")
            print('Classification Loss')
            if hparams['mixup']:
                print('Mixup')
            if hparams['enforce_prior']:
                print('Adversarial Loss')
                self.EPS = 1e-15
                self.D = Discriminator(self.z_size)
                self.D = self.D.to(self.device)
                self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.01, #!follow paper
                                              weight_decay=hparams['wd'])
            if hparams['metric_learning']:
                margin = hparams['metric_margin']
                metric_loss = hparams["metric_loss"]
                metric_weight = hparams["metric_weight"]
                print(
                    f"Metric Loss (margin: {margin} loss: {metric_loss} weight: {metric_weight})")
                self.M_optimizer = optim.SGD(
                    self.net.parameters(), momentum=0.9, lr=1e-3, weight_decay=1e-8)
                self.metric_weight = hparams['metric_weight']
                if metric_loss == 'random':
                    self.metric_loss = OnlineTripletLoss(
                        margin, RandomNegativeTripletSelector(margin))
                elif metric_loss == 'hardest':
                    self.metric_loss = OnlineTripletLoss(
                        margin, HardestNegativeTripletSelector(margin))
                elif metric_loss == 'semihard':
                    self.metric_loss = OnlineTripletLoss(
                        margin, SemihardNegativeTripletSelector(margin))
        # load model
        self.start_epoch = 0
        self.epoch = hparams['num_epochs']
        if hparams.get('restore',None) is not None:
            start_epoch, _ = self.load_model(hparams['restore'])
            self.start_epoch = hparams['num_epochs'] #tmp
    
    def _train(self, cur_epoch, trail_id, training=True):
        if training:
            self.net.train()
            self.net.training = True
        else:
            self.net.eval()
            self.net.training = False
        #follow ptbxl benchmark !!!
        train_losses = 0.0
        clf_losses = 0.0
        metric_losses = 0.0
        d_losses = 0.0
        g_losses = 0.0
        correct = 0
        total = 0
        n_batch = len(self.train_loader)
        confusion_matrix = torch.zeros(len(self.classes), len(self.classes))
        preds = []
        targets = []
        #print(f'\n=> Training Epoch #{cur_epoch}')
        for batch_idx, batch in enumerate(self.train_loader):
            #print('Batch: ',batch)
            inputs, labels = batch[0].float().to(self.device), batch[1].to(self.device)
            #print('Input shape: ',inputs.shape)
            #print('Input mean ',inputs.mean())
            #print('Label: ',labels)
            seed_features = self.net.extract_features(inputs)
            features = seed_features
            if self.hparams['manifold_mixup']:
                features, targets_a, targets_b, lam = mixup_data(features, labels,
                                                                 0.2, use_cuda=True)
                features, targets_a, targets_b = map(Variable, (features,
                                                                targets_a, targets_b))
            # apply pba transformation
            if self.hparams['use_modals']:
                try:
                    features = self.pm.apply_policy(
                        features, labels, cur_epoch, batch_idx, verbose=1).to(self.device)
                except Exception as e:
                    print(e)
                    print('tmp ignore error')
            #print('Feature shape: ',features.shape)
            outputs = self.net.classify(features)  # Forward Propagation
            #print('Output: ',outputs)
            if self.hparams['mixup']:
                inputs, targets_a, targets_b, lam = mixup_data(outputs, labels,
                                                               self.hparams['alpha'], use_cuda=True)
                inputs, targets_a, targets_b = map(Variable, (outputs,
                                                              targets_a, targets_b))
            # freeze D
            if self.hparams['enforce_prior']:
                for p in self.D.parameters():
                    p.requires_grad = False
            # classification loss
            if self.hparams['mixup'] or self.hparams['manifold_mixup']:
                c_loss = mixup_criterion(
                    self.criterion, outputs, targets_a, targets_b, lam)
            else:
                if self.multilabel:
                    c_loss = self.criterion(outputs, labels.float())  # Loss
                else:
                    c_loss = self.criterion(outputs, labels.long())  # Loss
            clf_losses += c_loss.item()
            # total loss
            loss = c_loss
            #print('Loss: ',loss)
            if self.hparams['metric_learning']:
                m_loss = self.metric_loss(seed_features, labels)[0]
                metric_losses += m_loss.item()
                loss = self.metric_weight * m_loss + \
                    (1-self.metric_weight) * c_loss
            train_losses += loss.item()

            if self.hparams['enforce_prior']:
                # Regularizer update
                # freeze D
                for p in self.D.parameters():
                    p.requires_grad = False
                self.net.train()
                d_fake = self.D(features)
                g_loss = self.hparams['prior_weight'] * \
                    adverserial_loss(d_fake, self.EPS)
                g_losses += g_loss.item()
                loss += g_loss

            self.optimizer.zero_grad()
            if training:
                loss.backward()  # Backward Propagation
                clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.optimizer.step()  # Optimizer update
                try: #tmp
                    self.scheduler.step()
                except Exception as e:
                    print('Exception:')
                    print(e)

            if self.hparams['enforce_prior']:
                # Discriminator update
                for p in self.D.parameters():
                    p.requires_grad = True

                features = self.net.extract_features(inputs)
                d_real = self.D(torch.randn(features.size()).to(self.device))
                d_fake = self.D(F.softmax(features, dim=0))
                d_loss = discriminator_loss(d_real, d_fake, self.EPS)
                self.D_optimizer.zero_grad()
                d_loss.backward()
                self.D_optimizer.step()
                d_losses += d_loss.item()

            # Accuracy / AUROC
            if not self.multilabel:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if self.hparams['mixup']:
                    correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                            + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
                else:
                    correct += (predicted == labels).sum().item()
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            else:
                predicted = torch.sigmoid(outputs).cpu().detach()
                if torch.any(torch.isnan(predicted)):
                    print('Inputs:', inputs.shape)
                    print(torch.max(inputs))
                    print(torch.min(inputs))
                    print('Representation: ', features.shape)
                    print(features)
                    print(torch.min(features))
                    print('Labels: ', labels.shape)
                    print(labels)
                    print(predicted)
                    print(outputs)
                    assert False
            preds.append(predicted)
            targets.append(labels.cpu().detach())
        
        if not self.multilabel:
            perfrom = 100 * correct/total
            perfrom_cw = 100 * confusion_matrix.diag()/(confusion_matrix.sum(1)+1e-9)
        else:
            targets_np = torch.cat(targets).numpy()
            preds_np = torch.cat(preds).numpy()
            try:
                perfrom = 100 * roc_auc_score(targets_np, preds_np,average='macro')
            except Exception as e:
                print('target shape: ',targets_np.shape)
                print('preds shape: ',preds_np.shape)
                nan_count = np.sum(np.isnan(preds_np))
                inf_count = np.sum(np.isinf(preds_np))
                print('predict nan, inf count: ',nan_count,inf_count)
                print(np.sum(targets_np,axis=0))
                print(np.sum(preds_np,axis=0))
                print(preds_np[np.isnan(preds_np)])
                raise e
            perfrom_cw = AUROC_cw(targets_np,preds_np)
            perfrom = perfrom_cw.mean()
        epoch_loss = train_losses/n_batch
        # step
        step = (cur_epoch-1)*(len(self.train_loader)) + batch_idx
        total_steps = self.hparams['num_epochs']*len(self.train_loader)
        #wandb dic
        out_dic = {}
        out_dic[f'train_loss'] = epoch_loss
        out_dic[f'train_clfloss'] = clf_losses/n_batch
        # logs
        display = f'| Epoch [{cur_epoch}/{self.hparams["num_epochs"]}]\tIter[{step}/{total_steps}]\tLoss: {epoch_loss:.4f}\tAcc@1/MacromAP: {perfrom:.4f}\tclf_loss: {clf_losses/n_batch:.4f}'
        if self.hparams['enforce_prior']:
            display += f'\td_loss: {d_losses/n_batch:.4f}\tg_loss: {g_losses/n_batch:.4f}'
            out_dic[f'train_d_loss'] = d_losses/n_batch
            out_dic[f'train_g_loss'] = g_losses/n_batch
        if self.hparams['metric_learning']:
            display += f'\tmetric_loss: {metric_losses/n_batch:.4f}'
            out_dic[f'train_metric_loss'] = metric_losses/n_batch
        print(display)
        if self.multilabel:
            ptype = 'auroc'
        else:
            ptype = 'acc'
        out_dic[f'train_{ptype}_avg'] = perfrom
        for i,e_c in enumerate(perfrom_cw):
            out_dic[f'train_{ptype}_c{i}'] = e_c

        return perfrom, epoch_loss, out_dic

    def _test(self, cur_epoch, trail_id, mode):
        self.net.eval()
        self.net.training = False
        correct = 0
        total = 0
        test_loss = 0.0
        if mode == 'train':
            data_loader = self.train_loader
        elif mode == 'valid':
            data_loader = self.valid_loader
        else:
            data_loader = self.test_loader
        confusion_matrix = torch.zeros(len(self.classes), len(self.classes))
        preds = []
        targets = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs, labels = batch[0].float().to(self.device), batch[1].to(self.device)

                outputs = self.net(inputs)
                if self.multilabel:
                    loss = self.criterion(outputs, labels.float())  # Loss multilabel
                else:
                    loss = self.criterion(outputs, labels.long())  # Loss singlelabel
                test_loss += loss.item()

                if not self.multilabel:
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                else:
                    predicted = torch.sigmoid(outputs.data)
                preds.append(predicted.cpu().detach())
                targets.append(labels.cpu().detach().long())
                
                torch.cuda.empty_cache()
        #prediction output
        output_dic = {}
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
        output_dic[f'{mode}_target'] = targets_np
        output_dic[f'{mode}_predict'] = preds_np
        #class-wise
        if not self.multilabel:
            perfrom = 100 * correct/total
            perfrom_cw = 100 * confusion_matrix.diag() / (confusion_matrix.sum(1)+1e-9)
        else:
            perfrom_cw = AUROC_cw(targets_np,preds_np)
            perfrom = perfrom_cw.mean()
        epoch_loss = test_loss / len(data_loader)

        if not self.multilabel:
            print(f'| ({mode}) Epoch #{cur_epoch}\t Loss: {epoch_loss:.4f}\t Acc@1: {perfrom:.4f}')
            print(f'class-wise Acc: ','['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        else:
            print(f'| ({mode}) Epoch #{cur_epoch}\t Loss: {epoch_loss:.4f}\t MacroAUROC: {perfrom:.4f}')
            print(f'class-wise AUROC: ','['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        #wandb dic
        out_dic = {}
        out_dic[f'{mode}_loss'] = epoch_loss
        if self.multilabel:
            ptype = 'auroc'
        else:
            ptype = 'acc'
        out_dic[f'{mode}_{ptype}_avg'] = perfrom
        for i,e_c in enumerate(perfrom_cw):
            out_dic[f'{mode}_{ptype}_c{i}'] = e_c

        return perfrom, epoch_loss, out_dic, output_dic
