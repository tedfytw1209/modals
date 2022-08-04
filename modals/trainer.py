import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networks.blstm import BiLSTM,LSTM
from networks.LSTM import LSTM_ecg,LSTM_ptb
from networks.LSTM_attention import LSTM_attention
from networks.Sleep_stager import SleepStagerChambon2018
from networks.resnet1d import resnet1d_wang
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from utility import mixup_criterion, mixup_data, mAP_cw, AUROC_cw

from modals.data_util import get_text_dataloaders,get_ts_dataloaders
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
from modals.operation_tseries import ToTensor,TransfromAugment,TS_OPS_NAMES,TS_ADD_NAMES,ECG_OPS_NAMES
import wandb

def count_parameters(model):
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f' |Trainable parameters: {temp}')


def build_model(model_name, vocab, n_class, z_size=2, dataset=''):
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
        n_hidden = 256
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
                  'ps_head': 0.5}
        net = resnet1d_wang(config)
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
    else:
        ValueError(f'Invalid model name={model_name}')

    print('\n### Model ###')
    print(f'=> {model_name}')
    print(f'embedding=> {z_size}')
    count_parameters(net)

    return net, z_size, model_name


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
            clip_grad_norm_(self.net.parameters(), 5.0)
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
            if self.hparams['mixup']:
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
        path = os.path.join(
            ckpt_dir, self.hparams['dataset_name'], f'{self.name}_{self.file_name}{title}')
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
        #wandb.config.update(hparams)
        self.name = name
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
        random.seed(0)
        self.train_loader, self.valid_loader, self.test_loader, self.classes, self.vocab = get_ts_dataloaders(
            hparams['dataset_name'], valid_size=hparams['valid_size'], batch_size=hparams['batch_size'],
            subtrain_ratio=hparams['subtrain_ratio'], dataroot=hparams['dataset_dir'],multilabel=self.multilabel,
            default_split=hparams['default_split'],labelgroup=hparams['labelgroup'],randaug_dic=self.randaug_dic,
            fix_policy_list=fix_policy
            )
        random.seed()
        self.device = torch.device(
            hparams['gpu_device'] if torch.cuda.is_available() else 'cpu')
        print()
        print('### Device ###')
        print(self.device)
        self.net, self.z_size, self.file_name = build_model(hparams['model_name'], self.vocab, len(self.classes))
        self.net = self.net.to(self.device)
        if self.multilabel:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.criterion = nn.CrossEntropyLoss()
        if hparams['mode'] in ['train', 'search']:
            self.optimizer = optim.AdamW(self.net.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['wd']) #follow ptbxl batchmark
            self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.hparams['lr'], epochs = self.hparams['num_epochs'], steps_per_epoch = len(self.train_loader))
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
    
    def _train(self, cur_epoch, trail_id):
        self.net.train()
        self.net.training = True
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
            loss.backward()  # Backward Propagation
            clip_grad_norm_(self.net.parameters(), 5.0)
            self.optimizer.step()  # Optimizer update
            self.scheduler.step()

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
            perfrom_cw = 100 * confusion_matrix.diag()/confusion_matrix.sum(1)
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
        
        if not self.multilabel:
            perfrom = 100 * correct/total
            perfrom_cw = 100 * confusion_matrix.diag() / (confusion_matrix.sum(1)+1e-12)
        else:
            targets_np = torch.cat(targets).numpy()
            preds_np = torch.cat(preds).numpy()
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

        return perfrom, epoch_loss, out_dic

    def run_model(self, epoch, trail_id):
        if self.hparams['use_modals']:
            self.pm.reset_tseries_data_pool(
                self.net, self.train_loader, self.hparams['temperature'], self.hparams['distance_metric'], self.hparams['dataset_name'])

        train_acc, tl, train_dic = self._train(epoch, trail_id)
        self.loss_dict['train'].append(tl)

        if self.hparams['valid_size'] > 0:
            val_acc, vl,val_dic = self._test(epoch, trail_id, mode='valid')
            self.loss_dict['valid'].append(vl)
            train_dic.update(val_dic)
        else:
            val_acc = 0.0

        return train_acc, val_acc, train_dic
    #change augment
    def change_augment(self,new_m):
        #not good code !!!
        print('Setting new augment m to ', new_m)
        new_augment = [
            ToTensor(),
            TransfromAugment(self.fix_policy,new_m,n=self.hparams['rand_n'],p=self.hparams['aug_p'])
            ]
        self.train_loader.dataset.augmentations = new_augment
        self.valid_loader.dataset.augmentations = new_augment
        self.test_loader.dataset.augmentations = new_augment