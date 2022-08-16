from modals.trainer import TextModelTrainer,TSeriesModelTrainer
from modals.setup import create_parser, create_hparams
import wandb
import time
import ray
import ray.tune as tune
from ray.tune.schedulers import PopulationBasedTraining,ASHAScheduler
from ray.tune.integration.wandb import WandbTrainableMixin
from ray.tune.schedulers import PopulationBasedTrainingReplay

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

def main(FLAGS, hparams):
    if FLAGS.kfold==10: #not ready
        print(f'Running 10 fold result')
        #hparams['kfold'] = tune.grid_search([i for i in range(hparams['kfold'])])
    elif FLAGS.kfold>=0:
        print(f'Running fold {FLAGS.kfold}/10 result')
    #trainer = TSeriesModelTrainer(hparams, FLAGS.name)
    #wandb
    if FLAGS.use_modals:
        Aug_type = 'MODAL'
        proj = 'MODAL'
    elif FLAGS.mixup:
        Aug_type = 'MIXUP'
        proj = 'MIXUP'
    elif FLAGS.manifold_mixup:
        Aug_type = 'MANIFOLDMIXUP'
        proj = 'MIXUP'
    elif FLAGS.randaug:
        if isinstance(FLAGS.rand_m,list):
            FLAGS.rand_m = FLAGS.rand_m[0]
        Aug_type = f'RANDAUG_{FLAGS.rand_m}_{FLAGS.rand_n}'
        proj = 'RandAugment'
    elif FLAGS.fix_policy!=None:
        if FLAGS.class_wise:
            Aug_type = 'Transfrom_classwise'
        else:
            Aug_type = FLAGS.fix_policy
        proj = 'RandAugment'
    else:
        Aug_type = 'NOAUG'
        if FLAGS.metric_learning:
            proj = 'MODAL'
        else:
            proj = 'RandAugment'
    experiment_name = f'{Aug_type}_train_{FLAGS.dataset}{FLAGS.labelgroup}_{FLAGS.model_name}_e{FLAGS.epochs}_lr{FLAGS.lr}'
    '''run_log = wandb.init(config=FLAGS, 
                  project=proj,
                  group=experiment_name,
                  name=f'{now_str}_' + experiment_name,
                  dir='./',
                  job_type="DataAugment",
                  reinit=True)'''
    hparams['restore'] = FLAGS.restore
    #change to ray model
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
    
    ray.init()
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

if __name__ == "__main__":
    FLAGS = create_parser('train')
    hparams = create_hparams('train', FLAGS)
    main(FLAGS, hparams)
