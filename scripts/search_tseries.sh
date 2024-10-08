if [[ $1 = "ptbxl" ]]; then
    CUDA_VISIBLE_DEVICES=3,4 \
    python search_tseries.py \
        --model_name lstm_ptb \
        --data_dir /mnt/data2/teddy/ptbxl-dataset \
        --dataset ptbxl \
        --subtrain_ratio 1.0 \
        --valid_size 1 \
        --default_split \
        --epochs 50 \
	    --bs 128 \
	    --lr 0.01 \
        --wd 0.01 \
        --gpu 0.25 --cpu 2 \
        --num_samples 16 --perturbation_interval 3  \
        --ray_name ray_experiment_ptbxl \
        --distance_metric loss \
        --metric_learning \
        --metric_loss random \
        --metric_weight 0.03 \
        --metric_margin 0.5 \
        --enforce_prior \
        --prior_weight 1 \
        --labelgroup subdiagnostic
elif [[ $1 = "wisdm" ]]; then
    CUDA_VISIBLE_DEVICES=5 python search_tseries.py --model_name lstm_atten --data_dir /mnt/data2/teddy/modals-main/modals/datasets/wisdm-dataset --dataset wisdm --subtrain_ratio 1.0 --valid_size 1 \
        --epochs 400 --bs 128 --lr 0.001 --gpu 0.15 --cpu 2 --num_samples 16 --perturbation_interval 3 --ray_name ray_experiment_wisdm --distance_metric loss --metric_learning \
        --metric_loss random --metric_weight 0.03 --metric_margin 0.5 --enforce_prior --prior_weight 1 --default_split
fi
CUDA_VISIBLE_DEVICES=1     python search_tseries.py         --model_name resnet_wang         --data_dir /mnt/data2/teddy/ptbxl-dataset         --dataset ptbxl \
    --subtrain_ratio 1.0         --valid_size 1         --kfold 9         --epochs 50     --bs 128     --lr 0.01         --wd 0.01         --gpu 0.25 --cpu 2 \
    --num_samples 16 --perturbation_interval 3          --ray_name ray_ptbsup_kfold_resnet_modal         --distance_metric loss         --metric_learning \
    --metric_loss random         --metric_weight 0.03         --metric_margin 0.5         --enforce_prior         --prior_weight 1         --labelgroup superdiagnostic

CUDA_VISIBLE_DEVICES=6     python search_tseries.py         --model_name lstm_ptb         --data_dir /mnt/data2/teddy/ptbxl-dataset         --dataset ptbxl \
 --subtrain_ratio 1.0         --valid_size 1         --default_split         --epochs 50     --bs 128     --lr 0.01         --wd 0.01         --gpu 0.25 --cpu 2 \
 --ray_name ray_experiment_ptbxl --labelgroup subdiagnostic --randaug --rand_m 0.1 0.2 0.3 0.4 0.5 --rand_n 1 2 3 4 5

CUDA_VISIBLE_DEVICES=0,1 python experiment_tseries.py --model_name lstm_ptb --data_dir /mnt/data2/teddy/ptbxl-dataset --dataset ptbxl --subtrain_ratio 1.0 --valid_size 1 \
 --default_split --epochs 50 --bs 128 --lr 0.01 --wd 0.01 --gpu 0.5 --cpu 4 --ray_name ray_ptbsuper_classdepend_lstm --labelgroup superdiagnostic --fix_policy exp_search \
 --rand_m 0 --rand_n 1 --aug_p 1.0 --num_m 50 --num_repeat 10 --info_region a,p,t,n --restore /mnt/data2/teddy/modals-main/checkpoints/ptbxl/ptbsuper_noaug_lstm_ptbbest

CUDA_VISIBLE_DEVICES=1,2,3,6 python search_tseries.py --model_name resnet_wang --dataset ptbxl --valid_size 1 --subtrain_ratio 1.0 \
 --epochs 50 --temperature 1 --bs 128 --lr 0.01 --wd 0.01 --data_dir /mnt/data2/teddy/ptbxl-dataset --kfold 9 --labelgroup subdiagnostic \
 --gpu 0.33 --cpu 2 --ray_name ray_ptbsub_kfold_resnet_transfrom_mag --fix_policy mag-search --rand_m 0.1 0.3 0.5 0.7 0.9