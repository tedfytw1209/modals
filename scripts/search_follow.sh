#!!!Need test metric margin value from {0.5,1,2,4,8} for complete reproduce
if [[ $1 = "trec" ]]; then
    CUDA_VISIBLE_DEVICES=3 \
    python search.py \
        --model_name blstm \
        --dataset trec \
        --valid_size 0 \
        --epochs 100 \
        --bs 100 \
        --gpu 0.15 --cpu 2 \
        --num_samples 16 --perturbation_interval 3  \
        --ray_name ray_experiment_trec \
        --distance_metric loss \
        --metric_learning \
        --metric_loss random \
        --metric_weight 0.03 \
        --metric_margin 0.5 \
        --enforce_prior \
        --prior_weight 1 \
        --lr 0.01
elif [[ $1 = "sst2" ]]; then
    CUDA_VISIBLE_DEVICES=3 \
    python search.py \
        --model_name blstm \
        --dataset sst2 \
        --valid_size 0 \
        --epochs 100 \
        --bs 100 \
        --gpu 0.15 --cpu 2 \
        --num_samples 16 --perturbation_interval 3  \
        --ray_name ray_experiment_sst2 \
        --distance_metric loss \
        --metric_learning \
        --metric_loss random \
        --metric_weight 0.03 \
        --metric_margin 2 \
        --enforce_prior \
        --prior_weight 1 \
        --lr 0.01
fi
