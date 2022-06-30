if [[ $1 = "trec" ]]; then
    CUDA_VISIBLE_DEVICES=3 \
    python -u train.py \
        --model_name blstm \
        --dataset trec \
        --valid_size 0 \
        --policy_epochs 100 \
        --epochs 100 \
        --name trec_model \
        --temperature 1 \
        --distance_metric loss \
        --enforce_prior \
        --prior_weight 1 \
        --metric_learning \
        --metric_loss random \
        --metric_weight 0.03 \
        --metric_margin 0.5 \
        --lr 0.01
elif [[ $1 = "sst2" ]]; then
    CUDA_VISIBLE_DEVICES=3 \
    python -u train.py \
        --model_name blstm \
        --dataset sst2 \
        --valid_size 0 \
        --policy_epochs 100 \
        --epochs 100 \
        --name trec_model \
        --temperature 1 \
        --distance_metric loss \
        --enforce_prior \
        --prior_weight 1 \
        --metric_learning \
        --metric_loss random \
        --metric_weight 0.03 \
        --metric_margin 0.5 \
        --lr 0.01
fi
