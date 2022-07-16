if [[ $1 = "ptbxl" ]]; then
    CUDA_VISIBLE_DEVICES=0     python -u train_tseries.py         --model_name lstm_ptb         --dataset ptbxl         --valid_size 1         --subtrain_ratio 1.0 \
    --policy_epochs 50         --epochs 50         --name ptbxl_model         --temperature 1         --distance_metric loss         --enforce_prior         --prior_weight 1 \
    --metric_learning --metric_loss random         --metric_weight 0.03         --metric_margin 0.5        --bs 128 --lr 0.01 --wd 0.01       \
    --data_dir /mnt/data2/teddy/ptbxl-dataset --default_split
elif [[ $1 = "wisdm" ]]; then
    CUDA_VISIBLE_DEVICES=2 python -u train_tseries.py --model_name lstm_atten --dataset wisdm --valid_size 1 --subtrain_ratio 1.0 \
    --bs 128 --epochs 400 --name wisdm_model --temperature 1 --distance_metric loss --enforce_prior --prior_weight 1 \
    --metric_learning --metric_loss random --metric_weight 0.03 --metric_margin 0.5 --lr 0.001 \
    --data_dir /mnt/data2/teddy/modals-main/modals/datasets/wisdm-dataset --default_split
elif [[ $1 = "edfx" ]]; then
    CUDA_VISIBLE_DEVICES=1 \
    python -u train_tseries.py \
        --model_name lstm \
        --dataset edfx \
        --valid_size 1 \
        --subtrain_ratio 1.0 \
        --policy_epochs 100 \
        --epochs 100 \
        --name wisdm_model \
        --temperature 1 \
        --distance_metric loss \
        --enforce_prior \
        --prior_weight 1 \
        --metric_learning \
        --metric_loss random \
        --metric_weight 0.03 \
        --metric_margin 0.5 \
        --lr 0.01 \
        --data_dir /mnt/data2/teddy/mne_data/
fi
