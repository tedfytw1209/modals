#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=32gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:3
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=bianjiang
#SBATCH --qos=bianjiang
#SBATCH --reservation=bianjiang

module load conda
conda activate caap

python -u train_image_fold.py --model_name resnet50 --dataset mimic_lt --valid_size 1 --subtrain_ratio 1.0 --policy_epochs 50 --epochs 50 \
 --name mimiclt_noaug --temperature 1 --bs 64 --lr 0.01 --wd 0.01  --data_dir /red/bianjiang/VLM_dataset/ReportGeneration/MIMIC-CXR_JPG/ \
 --kfold 10 --gpu 0.25 --cpu 2 --ray_name ray_mimiclt_noaug --multilabel