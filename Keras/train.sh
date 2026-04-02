#!/bin/bash
#SBATCH -p dgx # partition (queue). #SBATCH -w dgx2
#SBATCH --qos=quick
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=train
#SBATCH --time=24:00:00 # time (D-HH:MM)
#SBATCH --output=results.txt
#SBATCH --error=errors.txt

conda init bash
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate continuum-rl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

python -m Keras.DDPG --mode train
