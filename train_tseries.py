from modals.trainer import TextModelTrainer,TSeriesModelTrainer
from modals.setup import create_parser, create_hparams
import wandb

def main(FLAGS, hparams):
    start_epoch = 0
    trainer = TSeriesModelTrainer(hparams, FLAGS.name)
    trail_id = 0
    #wandb
    if FLAGS.use_modals:
        Aug_type = 'MODAL'
    elif FLAGS.mixup:
        Aug_type = 'MIXUP'
    elif FLAGS.manifold_mixup:
        Aug_type = 'MANIFOLDMIXUP'
    else:
        Aug_type = 'NOAUG'
    experiment_name = f'{Aug_type}_train_{FLAGS.dataset}_{FLAGS.model_name}_e{FLAGS.epochs}_lr{FLAGS.lr}'
    run_log = wandb.init(config=FLAGS, 
                  project='Myresearch',
                  name=experiment_name,
                  dir='./',
                  job_type="DataAugment",
                  reinit=True)
    if FLAGS.restore is not None:
        start_epoch, _ = trainer.load_model(FLAGS.restore)

    for e in range(start_epoch+1, hparams['num_epochs']+1):
        step_dic = {'epoch':e}
        train_acc, valid_acc, info_dict = trainer.run_model(e,trail_id)
        step_dic.update(info_dict)
        if e % 20 == 0:
            # print(hparams)
            trainer.save_checkpoint(hparams['checkpoint_dir'], e)
            test_acc, test_loss, info_dict_test = trainer._test(e, trail_id, 'test')
            step_dic.update(info_dict_test)
        wandb.log(step_dic)
    trainer.save_checkpoint(hparams['checkpoint_dir'], e)
    test_acc, test_loss, info_dict_test = trainer._test(hparams['num_epochs'], trail_id, 'test')
    step_dic.update(info_dict_test)
    wandb.log(step_dic)

if __name__ == "__main__":
    FLAGS = create_parser('train')
    hparams = create_hparams('train', FLAGS)
    main(FLAGS, hparams)
