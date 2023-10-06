#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=120GB
#SBATCH -o outputs-%j

export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/inspectEnv/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate inspectEnv

# wandb disabled 
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"

OUT_DIR=/scratch/general/vast/u1419542/cs6966/assignment4/models
# OUT_DIR=/scratch/general/vast/u1419542/cs6966/ass4/
mkdir -p ${OUT_DIR}

python3 model.py --output_dir ${OUT_DIR} -batchSize 1