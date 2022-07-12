if [[ $1 = "ptbxl" ]]; then
    CUDA_VISIBLE_DEVICES=3 \
    python search_tseries.py \
        --model_name lstm \
        --data_dir /mnt/data2/teddy/modals-main/modals/datasets/wisdm-dataset \
        --dataset ptbxl \
        --subtrain_ratio 1.0 \
        --valid_size 0.1 \
        --epochs 100 \
	    --bs 100 \
	    --lr 0.01 \
        --gpu 0.15 --cpu 2 \
        --num_samples 16 --perturbation_interval 3  \
        --ray_name ray_experiment_ptbxl \
        --distance_metric loss \
        --metric_learning \
        --metric_loss random \
        --metric_weight 0.03 \
        --metric_margin 0.5 \
        --enforce_prior \
        --prior_weight 1
elif [[ $1 = "wisdm" ]]; then
    CUDA_VISIBLE_DEVICES=5 python search_tseries.py --model_name lstm_atten --data_dir /mnt/data2/teddy/modals-main/modals/datasets/wisdm-dataset --dataset wisdm --subtrain_ratio 1.0 --valid_size 1 \
        --epochs 400 --bs 128 --lr 0.001 --gpu 0.15 --cpu 2 --num_samples 16 --perturbation_interval 3 --ray_name ray_experiment_wisdm --distance_metric loss --metric_learning \
        --metric_loss random --metric_weight 0.03 --metric_margin 0.5 --enforce_prior --prior_weight 1 --default_split
fi
