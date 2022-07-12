import random
import wandb
import numpy as np
import ray
import ray.tune as tune
from modals.setup import create_hparams, create_parser
from modals.trainer import TSeriesModelTrainer
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.wandb import WandbTrainableMixin


API_KEY = 'cb4c412d9f47cd551e38050ced659b0c58926986'

class RayModel(WandbTrainableMixin, tune.Trainable):
    def setup(self, *args): #use new setup replace _setup
        self.trainer = TSeriesModelTrainer(self.config)

    def step(self):#use step replace _train
        print(f'Starting Ray ID {self.trial_id} Iteration: {self._iteration}')
        step_dic = {f'epoch':self._iteration}
        train_acc, valid_acc, info_dict = self.trainer.run_model(self._iteration, self.trial_id)
        test_acc, test_loss, info_dict_test = self.trainer._test(self._iteration, self.trial_id, mode='test')
        step_dic.update(info_dict)
        step_dic.update(info_dict_test)
        wandb.log(step_dic)
        call_back_dic = {'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc}
        #call_back_dic.update(step_dic)
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


def search():
    FLAGS = create_parser('search')
    hparams = create_hparams('search', FLAGS)
    #wandb
    experiment_name = f'MODAL_search_{FLAGS.dataset}_{FLAGS.model_name}_e{FLAGS.epochs}_lr{FLAGS.lr}_ray{FLAGS.ray_name}'
    '''run_log = wandb.init(config=FLAGS, 
                  project='MODAL',
                  name=experiment_name,
                  dir='./',
                  job_type="DataAugment",
                  reinit=True)'''
    wandb_config = {
        'config':FLAGS, 
        'project':'MODAL',
        'group':experiment_name,
        #'name':experiment_name,
        'dir':'./',
        'job_type':"DataAugment",
        'reinit':True,
        'api_key':API_KEY,
    }
    hparams['wandb'] = wandb_config

    # if FLAGS.restore:
    #     train_spec["restore"] = FLAGS.restore

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

    ray.init()

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=FLAGS.perturbation_interval,
        custom_explore_fn=explore,
        log_config=True)

    analysis = tune.run(
        RayModel,
        name=hparams['ray_name'],
        scheduler=pbt,
        reuse_actors=True,
        verbose=True,
        metric="valid_acc",
        mode='max',
        checkpoint_score_attr="valid_acc",
        #checkpoint_freq=FLAGS.checkpoint_freq,
        resources_per_trial={"gpu": FLAGS.gpu, "cpu": FLAGS.cpu},
        stop={"training_iteration": hparams['num_epochs']},
        config=hparams,
        local_dir=FLAGS.ray_dir,
        num_samples=FLAGS.num_samples,
    )
    print("Best hyperparameters found were: ")
    print(analysis.best_config)
    print(analysis.best_trial)

    wandb.finish()

if __name__ == "__main__":
    search()
