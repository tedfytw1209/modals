import argparse
import random

from modals.augmentation_transforms import NUM_HP_TRANSFORM

RAY_DIR = './ray_results'
DATA_DIR = '/mnt/data2/teddy/textdata/'
EMB_DIR = '/mnt/data2/teddy/emb_dir/'
CP_DIR = './checkpoints'


def create_parser(mode):
    """Create arg parser for flags."""
    parser = argparse.ArgumentParser()

    ## Datasetting
    parser.add_argument('--data_dir', default=DATA_DIR, help='Directory where dataset is located.')
    parser.add_argument('--dataset', default='trec')
    parser.add_argument('--labelgroup', default='')
    parser.add_argument('--valid_size', type=int, default=500, help='Number of validation examples.')
    parser.add_argument('--subtrain_ratio', type=float, default=1.0, help='Ratio of sub training set')
    parser.add_argument('--default_split', action='store_true', help='use dataset deault split')

    ## Model and training setting
    parser.add_argument('--model_name',default='wrn')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0005,  help='weight decay')
    parser.add_argument('--bs', type=int, default=100, help='batch size')
    parser.add_argument('--gpu_device',  type=str, default='cuda:0')
    parser.add_argument('--multilabel',  action='store_true', help='otherwise use normal classification')

    parser.add_argument('--checkpoint_freq', type=int, default=50, help='Checkpoint frequency.')
    parser.add_argument('--checkpoint_dir', type=str, default=CP_DIR,  help='checkpoint directory.')
    parser.add_argument('--restore', type=str, default=None, help='If specified, tries to restore from given path.')

    ## Custom Modifications
    parser.add_argument('--temperature', type=float, default=1, help='temperature')
    parser.add_argument('--enforce_prior', action='store_true', help='otherwise use no policy')
    parser.add_argument('--prior_weight', type=float, default=1, help='weight of prior loss')
    parser.add_argument('--distance_metric', type=str, default='l2', help='metric used to weight the supporting samples', choices=('l2', 'loss', 'same', 'cosine'))
    parser.add_argument('--policy_path', type=str, default=None, help='text file storing a policy')
    parser.add_argument('--metric_learning', action='store_true', help='use metric learning')
    parser.add_argument('--metric_loss', type=str, default='random_triplets', help='type of triplet loss', choices=('semihard', 'random', 'hardest'))
    parser.add_argument('--metric_margin', type=float, default=1.0, help='metric margin')
    parser.add_argument('--metric_weight', type=float, default=0.5, help='weight of metric loss')

    parser.add_argument('--mixup', action='store_true', help='mixup benchmark')
    parser.add_argument('--alpha', type=float, default=1.0, help='mixup parameter')
    parser.add_argument('--manifold_mixup', action='store_true', help='manifold mixup benchmark')
    parser.add_argument('--randaug', action='store_true', help='RandAugment benchmark')
    parser.add_argument('--augselect', type=str, default='', help='RandAugment select data augment')
    parser.add_argument('--fix_policy', type=str, default=None, help='either a comma separated list of values')
    if mode == 'train':
        parser.add_argument('--use_modals', action='store_true', help='otherwise use no policy')
        parser.add_argument('--hp_policy', type=str, default=None, help='either a comma separated list of values')
        parser.add_argument('--policy_epochs', type=int, default=200, help='number of epochs/iterations policy trained for')
        parser.add_argument('--name', type=str, default='autoaug')
        parser.add_argument('--rand_m', type=float, default=0.5, help='RandAugment parameter m: Magnitude for all the transformations')
        parser.add_argument('--rand_n', type=int, default=1, help='RandAugment parameter n: Number of augmentation transformations')

    elif mode == 'search':
        ## Ray setting
        parser.add_argument('--ray_dir', type=str, default=RAY_DIR,  help='Ray directory.')
        parser.add_argument('--num_samples', type=int, default=3, help='Number of Ray samples')
        parser.add_argument('--cpu', type=float, default=4, help='Allocated by Ray')
        parser.add_argument('--gpu', type=float, default=0.12, help='Allocated by Ray')
        parser.add_argument('--perturbation_interval', type=int, default=3)
        parser.add_argument('--ray_name', type=str, default='ray_experiment')
        parser.add_argument('--ray_replay', type=str, default='', help='ray replay pbt')
        parser.add_argument('--rand_m',type=float, nargs='+', default=0.5, help='RandAugment parameter m: Magnitude for all the transformations')
        parser.add_argument('--rand_n',type=int, nargs='+', default=1, help='RandAugment parameter n: Number of augmentation transformations')
        
    else:
        raise ValueError('unknown state')

    return parser.parse_args()


def create_hparams(mode, FLAGS):
    """Creates hyperparameters to pass into Ray config.

  Different options depending on search or eval mode.

  Args:
    mode: a string, 'train' or 'test' or 'search'.
    FLAGS: parsed command line flags.

  Returns: dict
  """
    if FLAGS.randaug:
        print('RandAugment Parameters:')
        print(FLAGS.rand_m)
        print(FLAGS.rand_n)
    hparams = {
        'valid_size': FLAGS.valid_size,
        'dataset_name': FLAGS.dataset,
        'dataset_dir': FLAGS.data_dir,
        'checkpoint_dir': FLAGS.checkpoint_dir,
        'batch_size': FLAGS.bs,
        'multilabel': FLAGS.multilabel,
        'gradient_clipping_by_global_norm': 5.0,
        'mixup': FLAGS.mixup,
        'alpha': FLAGS.alpha,
        'lr': FLAGS.lr,
        'wd': FLAGS.wd,
        'momentum': 0.9,
        'gpu_device': FLAGS.gpu_device,
        'mode': mode,
        'temperature': FLAGS.temperature if FLAGS.temperature<=1 else FLAGS.temperature/10,
        'distance_metric': FLAGS.distance_metric,
        'enforce_prior': FLAGS.enforce_prior,
        'metric_learning': FLAGS.metric_learning,
        'subtrain_ratio': FLAGS.subtrain_ratio, ## for text data controlling ratio of training data
        'manifold_mixup': FLAGS.manifold_mixup,
        'multilabel': FLAGS.multilabel,
        'default_split': FLAGS.default_split,
        'labelgroup': FLAGS.labelgroup,
        'randaug': FLAGS.randaug,
        'augselect': FLAGS.augselect,
        'fix_policy': FLAGS.fix_policy,
        }

    if FLAGS.enforce_prior:
        hparams['prior_weight'] = FLAGS.prior_weight if FLAGS.prior_weight<=1 else FLAGS.prior_weight/10

    if FLAGS.metric_learning:
        hparams['metric_loss'] = FLAGS.metric_loss
        hparams['metric_margin'] = FLAGS.metric_margin
        hparams['metric_weight'] = FLAGS.metric_weight

    if mode == 'train':
        hparams['use_modals'] = FLAGS.use_modals
        hparams['policy_path'] = None
        hparams['hp_policy'] = None
        hparams['fix_policy'] = FLAGS.fix_policy
        if FLAGS.use_modals:
            if FLAGS.hp_policy == 'random':
                # random policy
                parsed_policy = [random.randrange(0, 11) for i in range(NUM_HP_TRANSFORM * 4)]
                hparams['hp_policy'] = parsed_policy
            elif FLAGS.hp_policy == 'average':
                # random policy
                parsed_policy = [5]* (NUM_HP_TRANSFORM * 4)
                hparams['hp_policy'] = parsed_policy
            elif FLAGS.policy_path is not None:
                # supplied a schedule
                hparams['policy_path'] = FLAGS.policy_path
            else:
                # parse input into a fixed augmentation policy
                parsed_policy = FLAGS.hp_policy.split(',')
                parsed_policy = [int(p) for p in parsed_policy]
                hparams['hp_policy'] = parsed_policy
        if FLAGS.randaug or FLAGS.fix_policy:
            hparams['rand_m'] = FLAGS.rand_m
            hparams['rand_n'] = FLAGS.rand_n
    elif mode == 'search':
        if FLAGS.randaug:
            #RandAug search
            hparams['use_modals'] = False
            hparams['rand_m'] = FLAGS.rand_m
            hparams['rand_n'] = FLAGS.rand_n
        elif 'search' in FLAGS.fix_policy:
            hparams['use_modals'] = False
            hparams['rand_m'] = FLAGS.rand_m
            hparams['rand_n'] = 1
        else: #MODAL search
            hparams['use_modals'] = True
            hparams['policy_path'] = None
            # default start value of 0
            hparams['hp_policy'] = [0 for _ in range(4 * NUM_HP_TRANSFORM)]

        hparams['ray_name']  = FLAGS.ray_name

    else:
        raise ValueError('unknown mode')

    # Child model
    hparams['model_name'] = FLAGS.model_name

    # epochs is put here for later setting for specific models
    hparams['num_epochs'] = FLAGS.epochs

    return hparams
