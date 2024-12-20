#!/bin/bash -l
#SBATCH -J pytestjob   #job name
#SBATCH --time=00-48:00:00  #requested time (DD-HH:MM:SS)
#SBATCH --mem=16g  #requesting 16GB of RAM total for the number of cpus you requested
#SBATCH --partition=gpu #running on "gpu" partition
#SBATCH --gres=gpu:a100:2  #requesting 2 A100 GPUs, in this case, the "-p" needs to be switched to a partition that has the requested GPU resources
#SBATCH --constraint="a100-80G"
#SBATCH --exclude=s1cmp003

#SBATCH --output=pytestjob.%j.%N.out  #saving standard output to file -- %j jobID -- %N nodename
#SBATCH --error=pytestjob.%j.%N.err   #saving standard error to file -- %j jobID -- %N nodename
#SBATCH --mail-type=ALL
#SBATCH --mail-user=derek.oconnor@tufts.edu

echo "Loading anaconda and CUDA"
module load anaconda/2024.06-py312
module load cuda/11.7

echo "Initializing conda shell environment"
eval "$(conda shell.bash hook)" 

# echo "Activating aiml environment"
# conda activate aiml
pip install "numpy<2"

# Check GPU availability
echo "Checking GPU status with nvidia-smi"
nvidia-smi  # This will show the GPU status assigned to your job

echo "--- Running python script dnn_weight_noise.py ---"
python -u dnn_weight_noise.py --experiment-version v22 --experiment-type dnn --cloud-environment hpc --debug-mode experiment