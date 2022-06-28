if [[ $1 = "ptbxl" ]]; then
    CUDA_VISIBLE_DEVICES=3 \
    python search_tseries.py \
        --model_name blstm \
        --data_dir /mnt/data2/teddy/modals-main/modals/datasets/wisdm-dataset \
        --dataset trec \
        --subtrain_ratio 1.0 \
        --valid_size 0 \
        --epochs 100 \
	    --bs 100 \
	    --lr 0.01 \
        --gpu 0.15 --cpu 2 \
        --num_samples 16 --perturbation_interval 3  \
        --ray_name ray_experiment_trec \
        --distance_metric loss \
        --metric_learning \
        --metric_loss random \
        --metric_weight 0.03 \
        --metric_margin 0.5 \
        --enforce_prior \
        --prior_weight 1
elif [[ $1 = "wisdm" ]]; then
    CUDA_VISIBLE_DEVICES=3 \
    python search_tseries.py \
        --model_name blstm \
        --data_dir /mnt/data2/teddy/modals-main/modals/datasets/wisdm-dataset \
        --dataset wisdm \
	    --subtrain_ratio 1.0 \
        --valid_size 0 \
        --epochs 100 \
	    --bs 100 \
	    --lr 0.01 \
        --gpu 0.15 --cpu 2 \
        --num_samples 16 --perturbation_interval 3  \
        --ray_name ray_experiment_sst2 \
        --distance_metric loss \
        --metric_learning \
        --metric_loss random \
        --metric_weight 0.03 \
        --metric_margin 0.5 \
        --enforce_prior \
        --prior_weight 1
fi
