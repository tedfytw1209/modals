import random
import wandb
import numpy as np
import ray
import ray.tune as tune
from modals.setup import create_hparams, create_parser
from modals.trainer import TSeriesModelTrainer
from modals.operation_tseries import TS_OPS_NAMES,ECG_OPS_NAMES,TS_ADD_NAMES,MAG_TEST_NAMES,NOMAG_TEST_NAMES,EXP_TEST_NAMES
from ray.tune.schedulers import PopulationBasedTraining,ASHAScheduler
from ray.tune.integration.wandb import WandbTrainableMixin
from ray.tune.schedulers import PopulationBasedTrainingReplay
import time

now_str = time.strftime("%Y%m%d-%H%M%S")

API_KEY = 'cb4c412d9f47cd551e38050ced659b0c58926986'

def dict_append(result_dic,info_dict):
    for k in info_dict.keys():
        if result_dic.get(k,None)==None:
            result_dic[k] = [info_dict[k]]
        else:
            result_dic[k].append(info_dict[k])
def dict_avg(result_dic):
    output_dict = {}
    for k in result_dic.keys():
        output = np.array(result_dic[k])
        output_dict[f'{k}_avg'] = np.mean(output)
        output_dict[f'{k}_std'] = np.std(output)
    return output_dict

class RayModel(WandbTrainableMixin, tune.Trainable):
    def setup(self, *args): #use new setup replace _setup
        self.trainer = TSeriesModelTrainer(self.config)
        self.result_train_dic,self.result_valid_dic, self.result_test_dic = {}, {}, {}
        self.tmp_randm = self.config['rand_m'] #mag start at rand_m
        if self.config.get('restore',False):
            self._restore(self.config['restore'])

    def step(self):#use step replace _train
        if self._iteration==0:
            wandb.config.update(self.config)
        print(f'Starting Ray ID {self.trial_id} Iteration: {self._iteration}')
        #step_dic = {f'epoch':self._iteration}
        #train_acc, valid_acc, info_dict = self.trainer.run_model(self._iteration, self.trial_id)
        train_acc, train_loss, info_dict_train = self.trainer._test(self._iteration, self.trial_id, mode='train')
        valid_acc, valid_loss, info_dict_valid = self.trainer._test(self._iteration, self.trial_id, mode='valid')
        test_acc, test_loss, info_dict_test = self.trainer._test(self._iteration, self.trial_id, mode='test')
        dict_append(self.result_train_dic,info_dict_train)
        dict_append(self.result_valid_dic,info_dict_valid)
        dict_append(self.result_test_dic,info_dict_test)
        
        #if last epoch
        if self._iteration%self.config['num_repeat'] == self.config['num_repeat']-1:
            step_dic = {'m':self.tmp_randm}
            #average and var
            result_train = dict_avg(self.result_train_dic)
            result_valid = dict_avg(self.result_valid_dic)
            result_test = dict_avg(self.result_test_dic)
            step_dic.update(result_train)
            step_dic.update(result_valid)
            step_dic.update(result_test)
            wandb.log(step_dic)
            #clear
            self.result_train_dic={}
            self.result_test_dic={}
            self.result_valid_dic={}
            #reset data augment
            self.tmp_randm = min(self.tmp_randm + 1.0 / self.config['num_m'] , 1.0)
            self.trainer.change_augment(self.tmp_randm)
        call_back_dic = {'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc}
        return call_back_dic

    def _save(self, checkpoint_dir):
        print(checkpoint_dir)
        path = self.trainer.save_model(checkpoint_dir, self._iteration)
        print(path)
        return path

    def _restore(self, checkpoint_path):
        self.trainer.load_model(checkpoint_path)

    def reset_config(self, new_config):
        self.config = new_config
        self.trainer.reset_config(self.config)
        return True
    
def explore(config):
        """Custom explore function.
    Args:
      config: dictionary containing ray config params.
    Returns:
      Copy of config with modified augmentation policy.
    """
        new_params = []
        for i, param in enumerate(config["hp_policy"]):
            if random.random() < 0.2:
                new_params.append(random.randint(0, 10))
            else:
                amt = np.random.choice(
                    [0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
                amt = int(amt)
                if random.random() < 0.5:
                    new_params.append(max(0, param - amt))
                else:
                    new_params.append(min(10, param + amt))
        config["hp_policy"] = new_params
        return config

def search():
    FLAGS = create_parser('search')
    hparams = create_hparams('search', FLAGS)
    if FLAGS.randaug:
        method = 'RandAug'
        proj = 'RandAugment'
    elif 'search' in FLAGS.fix_policy:
        method = 'Transfrom'
        proj = 'RandAugment'
    else:
        print('error')
        exit()

    #wandb
    experiment_name = f'{now_str}_{method}_search{FLAGS.ray_replay}_{FLAGS.dataset}{FLAGS.labelgroup}_{FLAGS.model_name}_e{FLAGS.epochs}_lr{FLAGS.lr}_ray{FLAGS.ray_name}'
    '''run_log = wandb.init(config=FLAGS, 
                  project='MODAL',
                  name=experiment_name,
                  dir='./',
                  job_type="DataAugment",
                  reinit=True)'''
    wandb_config = {
        #'config':FLAGS, 
        'project':proj,
        'group':experiment_name,
        #'name':experiment_name,
        'dir':'./',
        'job_type':"DataAugment",
        'reinit':False,
        'api_key':API_KEY
    }
    #for wandb
    hparams["log_config"]= True
    hparams['wandb'] = wandb_config
    #for restore
    if FLAGS.restore:
        print('Using restore: ',FLAGS.restore)
        hparams["restore"] = FLAGS.restore

    ray.init()
    if FLAGS.randaug: #randaug search tune.grid_search from rand_m*rand_n
        total_grid = len(hparams['rand_m']) * len(hparams['rand_n'])
        print(f'RandAugment grid search for {total_grid} samples')
        hparams['rand_m'] = tune.grid_search(hparams['rand_m'])
        hparams['rand_n'] = tune.grid_search(hparams['rand_n'])
        #tune_scheduler = ASHAScheduler(metric="valid_acc", mode="max",max_t=hparams['num_epochs'],grace_period=10,
        #    reduction_factor=3,brackets=1)
        '''tune_scheduler = ASHAScheduler(max_t=hparams['num_epochs'],grace_period=25,
            reduction_factor=3,brackets=1)'''
        tune_scheduler = None
        analysis = tune.run(
            RayModel,
            name=hparams['ray_name'],
            scheduler=tune_scheduler,
            #reuse_actors=True,
            verbose=True,
            metric="valid_acc",
            mode='max',
            checkpoint_score_attr="valid_acc",
            #checkpoint_freq=FLAGS.checkpoint_freq,
            resources_per_trial={"gpu": FLAGS.gpu, "cpu": FLAGS.cpu},
            stop={"training_iteration": hparams['num_epochs']},
            config=hparams,
            local_dir=FLAGS.ray_dir,
            num_samples=1, #grid search no need
            )
    elif 'search' in FLAGS.fix_policy: #test over all transfroms
        total_epoch = hparams['num_epochs']
        if 'mag' in FLAGS.fix_policy:
            hparams['fix_policy'] = tune.grid_search(MAG_TEST_NAMES)
            hparams['rand_m'] = tune.grid_search(hparams['rand_m'])
            len_m = len(hparams['rand_m'])
        elif 'exp' in FLAGS.fix_policy:
            hparams['mode'] = 'test' #change mode to test
            hparams['fix_policy'] = tune.grid_search(EXP_TEST_NAMES)
            hparams['rand_m'] = hparams['rand_m'][0] #just for first m
            hparams['num_repeat'] = FLAGS.num_repeat
            hparams['num_m'] = FLAGS.num_m
            num_repeat = hparams['num_repeat']
            num_m = hparams['num_m']
            total_epoch = hparams['num_repeat'] * hparams['num_m']
            print(f'Each experiment search for {num_m} magnitudes and {num_repeat} samples')
            len_m = 1
        else:
            hparams['fix_policy'] = tune.grid_search(NOMAG_TEST_NAMES)
            hparams['rand_m'] = 0.5
            len_m = 1
        total_grid = len_m * len(hparams['fix_policy'])
        print('Using policy ',hparams['fix_policy'])
        print(f'Transfrom grid search for {total_grid} samples')
        
        tune_scheduler = None
        analysis = tune.run(
            RayModel,
            name=hparams['ray_name'],
            scheduler=tune_scheduler,
            #reuse_actors=True,
            verbose=True,
            metric="valid_acc",
            mode='max',
            checkpoint_score_attr="valid_acc",
            #checkpoint_freq=FLAGS.checkpoint_freq,
            resources_per_trial={"gpu": FLAGS.gpu, "cpu": FLAGS.cpu},
            stop={"training_iteration": total_epoch},
            config=hparams,
            local_dir=FLAGS.ray_dir,
            num_samples=1, #grid search no need
            )
    
    print("Best hyperparameters found were: ")
    print(analysis.best_config)
    print(analysis.best_trial)
    

    wandb.finish()
if __name__ == "__main__":
    search()
