#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=all_models.out

# Activate Anaconda work environment
source /home/kit/ksri/df6291/.bashrc
source activate af

# Iterate over the different models and pred lengths (96 336 192 720(720 is not working with batch 256!!!)),
# Autoformer, Transformer, AST
for MODEL in AST
do
  for LR in 0.001 # 0.0001
  do
    for LEN in 336 # 96
    do
#      python run.py --root_path data/ETT-small/ --data ETTm2 --model_id '' \
#        --model $MODEL --features M --freq t --seq_len 96 --label_len 48 --pred_len $LEN --e_layers 2 --d_layers 1 \
#        --factor 1 --enc_in 7 --dec_in 7 --c_out 7 --embed fixed --learning_rate $LR
#      python run.py --root_path data/electricity/ --data electricity --model_id '' \
#        --model $MODEL --features M --freq h --seq_len 96 --label_len 48 --pred_len $LEN --e_layers 2 --d_layers 1 \
#        --factor 1 --enc_in 321 --dec_in 321 --c_out 321 --embed fixed --learning_rate $LR
#      python run.py --root_path data/exchange_rate/ --data exchange_rate --model_id '' \
#        --model $MODEL --features M --freq d --seq_len 96 --label_len 48 --pred_len $LEN --e_layers 2 --d_layers 1 \
#        --factor 1 --enc_in 8 --dec_in 8 --c_out 8 --embed fixed --learning_rate $LR
      python run.py --root_path data/traffic/ --data traffic --model_id '' \
        --model $MODEL --features M --freq h --seq_len 96 --label_len 48 --pred_len $LEN --e_layers 2 --d_layers 1 \
        --factor 1 --enc_in 862 --dec_in 862 --c_out 862 --embed fixed --learning_rate $LR
      python run.py --root_path data/weather/ --data weather --model_id '' \
        --model $MODEL --features M --freq 10min --seq_len 96 --label_len 48 --pred_len $LEN --e_layers 2 --d_layers 1 \
        --factor 1 --enc_in 21 --dec_in 21 --c_out 21 --embed fixed --learning_rate $LR
    done
    # manual pred len required for illness dataset
    python run.py --root_path data/illness/ --data illness --model_id '' \
    --model $MODEL --features M --freq d --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 \
    --factor 1 --enc_in 7 --dec_in 7 --c_out 7 --embed fixed --learning_rate $LR
  done
done


# run script:
# sbatch -p sdil -t 1:00:00 all_models.sh

# for development:
# sbatch -p dev_gpu_4 -t 30 all_models.sh
