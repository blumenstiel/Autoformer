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
for MODEL in NST
do
  for LEN in 96
  do
    python run.py --data ETTm2 --model $MODEL --pred_len $LEN
    python run.py --data electricity --model $MODEL --pred_len $LEN
    python run.py --data exchange_rate --model $MODEL --pred_len $LEN
    python run.py --data traffic --model $MODEL --pred_len $LEN
    python run.py --data weather --model $MODEL --pred_len $LEN
  done
  # manual pred len required for illness dataset
  python run.py --data illness --model $MODEL --pred_len 24
done


# run script:
# sbatch -p sdil -t 3:00:00 all_models.sh

# for development:
# sbatch -p dev_gpu_4 -t 30 all_models.sh
