import random
from timeit import repeat
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
from ray.tune.suggest import Repeater
import time

now_str = time.strftime("%Y%m%d-%H%M%S")

API_KEY = 'cb4c412d9f47cd551e38050ced659b0c58926986'

class RayModel(WandbTrainableMixin, tune.Trainable):
    def setup(self, *args): #use new setup replace _setup
        self.trainer = TSeriesModelTrainer(self.config)
        self.result_valid_dic, self.result_test_dic = {}, {}

    def step(self):#use step replace _train
        if self._iteration==0:
            wandb.config.update(self.config)
        print(f'Starting Ray ID {self.trial_id} Iteration: {self._iteration}')
        step_dic = {f'epoch':self._iteration}
        train_acc, valid_acc, info_dict = self.trainer.run_model(self._iteration, self.trial_id)
        test_acc, test_loss, info_dict_test = self.trainer._test(self._iteration, self.trial_id, mode='test')
        self.result_valid_dic = {f'result_{k}': info_dict[k] for k in info_dict.keys()}
        self.result_test_dic = {f'result_{k}': info_dict_test[k] for k in info_dict_test.keys()}
        step_dic.update(info_dict)
        step_dic.update(info_dict_test)
        #if last epoch
        if self._iteration==self.config['num_epochs']-1:
            step_dic.update(self.result_valid_dic)
            step_dic.update(self.result_test_dic)
        wandb.log(step_dic)
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
    elif FLAGS.fix_policy and 'search' in FLAGS.fix_policy:
        method = 'Transfrom'
        proj = 'RandAugment'
    else:
        method = 'MODAL'
        proj = 'MODAL'
        now_str = ''
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
    hparams["log_config"]= False
    hparams['wandb'] = wandb_config

    # if FLAGS.restore:
    #     train_spec["restore"] = FLAGS.restore

    ray.init()
    if hparams['use_modals']: #MODALS search
        pbt = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=FLAGS.perturbation_interval,
            custom_explore_fn=explore,
            metric="valid_acc",
            mode='max',
            log_config=True)
        repeat_times = 1
        if FLAGS.kfold==10:
            print(f'Running 10 fold result')
            pbt = Repeater(pbt,repeat=FLAGS.kfold)
            repeat_times = 10
        elif FLAGS.kfold>=0:
            print(f'Running fold {FLAGS.kfold}/10 result')
        analysis = tune.run(
            RayModel,
            name=hparams['ray_name'],
            scheduler=pbt,
            reuse_actors=True,
            verbose=True,
            checkpoint_score_attr="valid_acc",
            #checkpoint_freq=FLAGS.checkpoint_freq,
            resources_per_trial={"gpu": FLAGS.gpu, "cpu": FLAGS.cpu},
            stop={"training_iteration": hparams['num_epochs']},
            config=hparams,
            local_dir=FLAGS.ray_dir,
            num_samples=FLAGS.num_samples*repeat_times,
            )
    elif FLAGS.randaug: #randaug search tune.grid_search from rand_m*rand_n
        total_grid = len(hparams['rand_m']) * len(hparams['rand_n'])
        print(f'RandAugment grid search for {total_grid} samples')
        hparams['rand_m'] = tune.grid_search(hparams['rand_m'])
        hparams['rand_n'] = tune.grid_search(hparams['rand_n'])
        if FLAGS.kfold==10:
            print(f'Running 10 fold result')
            hparams['kfold'] = tune.grid_search([i for i in range(hparams['kfold'])])
        elif FLAGS.kfold>=0:
            print(f'Running fold {FLAGS.kfold}/10 result')
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
        if 'mag' in FLAGS.fix_policy:
            hparams['fix_policy'] = tune.grid_search(MAG_TEST_NAMES)
            hparams['rand_m'] = tune.grid_search(hparams['rand_m'])
            len_m = len(hparams['rand_m'])
        elif 'fixmag-' in FLAGS.fix_policy:
            fix_policy = hparams['fix_policy'].split('-')[1]
            hparams['fix_policy'] = fix_policy
            hparams['rand_m'] = tune.grid_search(hparams['rand_m'])
            len_m = len(hparams['rand_m'])
        elif 'exp' in FLAGS.fix_policy:
            hparams['fix_policy'] = tune.grid_search(EXP_TEST_NAMES)
            total_aug = len(hparams['fix_policy'])
            hparams['num_m'] = FLAGS.num_m
            num_m = hparams['num_m']
            #make grid
            if len(hparams['rand_m'])==1:
                final_m = [hparams['rand_m'] for i in range(total_aug)]
            elif len(hparams['rand_m'])==total_aug:
                final_m = hparams['rand_m']
            else:
                print('Undefine rand_m, ERROR')
                raise
            grid_m = {EXP_TEST_NAMES[i]:np.linspace(0., final_m[i], num=num_m) for i in range(total_aug)}
            def resolve_randm(spec):
                fix_policy = spec.config.fix_policy
                randm_range = grid_m[fix_policy]
                return tune.grid_search(randm_range)
            hparams['rand_m'] = resolve_randm
            #printout
            print(f'Each experiment search for {num_m} magnitudes and {total_aug} transfroms')
            len_m = 1
        else:
            hparams['fix_policy'] = tune.grid_search(NOMAG_TEST_NAMES)
            hparams['rand_m'] = 0.5
            len_m = 1
        total_grid = len_m * len(hparams['fix_policy'])
        print(f'Transfrom grid search for {total_grid} samples')
        if FLAGS.kfold==10:
            print(f'Running 10 fold result')
            hparams['kfold'] = tune.grid_search([i for i in range(hparams['kfold'])])
        elif FLAGS.kfold>=0:
            print(f'Running fold {FLAGS.kfold}/10 result')
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
    wandb.finish()
    #print("Best hyperparameters found were: ")
    #print(analysis.best_config)
    #print(analysis.best_trial)
    

    
if __name__ == "__main__":
    search()
