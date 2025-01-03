from modals.trainer import ImageModelTrainer
from modals.setup import create_parser, create_hparams
import wandb
import time

now_str = time.strftime("%Y%m%d-%H%M%S")

def main(FLAGS, hparams):
    if FLAGS.kfold==10: #not ready mow!!!
        print(f'Running 10 fold result')
        #hparams['kfold'] = tune.grid_search([i for i in range(hparams['kfold'])])
    elif FLAGS.kfold>=0:
        print(f'Running fold {FLAGS.kfold}/10 result')
    start_epoch = 0
    trainer = ImageModelTrainer(hparams, FLAGS.name)
    trail_id = 0
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
    run_log = wandb.init(config=FLAGS, 
                  project=proj,
                  group=experiment_name,
                  name=f'{now_str}_' + experiment_name,
                  dir='./',
                  job_type="DataAugment",
                  reinit=True)
    if FLAGS.restore is not None:
        start_epoch, _ = trainer.load_model(FLAGS.restore)
    best_valid_acc,best_model = 0,None
    for e in range(start_epoch+1, hparams['num_epochs']+1):
        step_dic = {'epoch':e}
        train_acc, valid_acc, train_dic, val_output_dic = trainer.run_model(e,trail_id)
        info_dict = train_dic
        step_dic.update(info_dict)
        if e % 20 == 0:
            # print(hparams)
            trainer.save_checkpoint(hparams['checkpoint_dir'], e)
        test_acc, test_loss, info_dict_test, output_dic_test = trainer._test(e, trail_id, 'test')
        if valid_acc>best_valid_acc:
            best_valid_acc = valid_acc
            trainer.save_checkpoint(hparams['checkpoint_dir'], e,title='best')
            result_valid_dic = {f'result_{k}': info_dict[k] for k in info_dict.keys()}
            result_test_dic = {f'result_{k}': info_dict_test[k] for k in info_dict_test.keys()}
            step_dic['best_valid_acc_avg'] = valid_acc
            step_dic['best_test_acc_avg'] = test_acc
        step_dic.update(info_dict_test)
        wandb.log(step_dic)
    trainer.save_checkpoint(hparams['checkpoint_dir'], e)
    step_dic.update(result_valid_dic)
    step_dic.update(result_test_dic)
    wandb.log(step_dic)

if __name__ == "__main__":
    FLAGS = create_parser('train')
    hparams = create_hparams('train', FLAGS)
    main(FLAGS, hparams)
