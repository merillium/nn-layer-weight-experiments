#!/bin/bash -l
#SBATCH -J pytestjob   #job name
#SBATCH --time=00-96:00:00  #requested time (DD-HH:MM:SS)
#SBATCH --mem=16g  #requesting 16GB of RAM total for the number of cpus you requested
#SBATCH --partition=gpu #running on "gpu" partition
#SBATCH --gres=gpu:a100:2  #requesting 2 A100 GPUs, in this case, the "-p" needs to be switched to a partition that has the requested GPU resources
#SBATCH --constraint="a100-80G"

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

# Uncomment for training
# echo "--- Running python script dnn_weight_noise.py TRAINING ---"
# python -i dnn_weight_noise.py --experiment-version v37 --experiment-type dnn-train --learning-rates-dict "{'dnn2g':[1e-3,5e-4,1e-4], 'dnn2h':[1e-3,5e-4,1e-4]}" --cloud-environment hpc --debug-mode experiment
# python -i dnn_weight_noise.py --experiment-version v38 --experiment-type dnn-train --learning-rates-dict "{'dnn2i':[1e-3,5e-4,1e-4], 'dnn2j':[1e-3,5e-4,1e-4]}" --cloud-environment hpc --debug-mode experiment
# python -i dnn_weight_noise.py --experiment-version v39 --experiment-type dnn-train --learning-rates-dict "{'dnn1a':[1e-3,5e-4,1e-4], 'dnn1b':[1e-3,5e-4,1e-4], 'dnn1c':[1e-3,5e-4,1e-4], 'dnn1d':[1e-3,5e-4,1e-4]}" --cloud-environment hpc --debug-mode experiment
# python -i dnn_weight_noise.py --experiment-version v40 --experiment-type dnn-train --learning-rates-dict "{'dnn1e':[1e-3,5e-4,1e-4], 'dnn1f':[1e-3,5e-4,1e-4]}" --cloud-environment hpc --debug-mode experiment
# python -i dnn_weight_noise.py --experiment-version v41 --experiment-type dnn-train --learning-rates-dict "{'dnn1h':[1e-3,5e-4,1e-4], 'dnn1i':[1e-3,5e-4,1e-4]}" --cloud-environment hpc --debug-mode experiment
# python -i dnn_weight_noise.py --experiment-version v42 --experiment-type dnn-train --learning-rates-dict "{'dnn1j':[1e-3,5e-4,1e-4], 'dnn1k':[1e-3,5e-4,1e-4]}" --cloud-environment hpc --debug-mode experiment --normalization-type batch
# python -i dnn_weight_noise.py --experiment-version v43 --experiment-type dnn-train --learning-rates-dict "{'dnn2a':[1e-3,5e-4,1e-4], 'dnn2b':[1e-3,5e-4,1e-4], 'dnn2c':[1e-3,5e-4,1e-4], 'dnn2d':[1e-3,5e-4,1e-4]}" --cloud-environment hpc --debug-mode experiment --normalization-type batch

# python -i dnn_weight_noise.py --experiment-version v37 --experiment-type dnn-train --learning-rates-dict "{'dnn2h':[1e-4]}" --cloud-environment hpc --debug-mode experiment
# python -i dnn_weight_noise.py --experiment-version v38 --experiment-type dnn-train --learning-rates-dict "{'dnn2k':[1e-3,5e-4,1e-4]}" --cloud-environment local --debug-mode debug

# Uncomment for noise experiments
echo "--- Running python script dnn_weight_noise.py NOISE EXPERIMENT ---"
# python -i dnn_weight_noise.py --experiment-version v35 --experiment-type dnn-load-model --pretrained-model-name "dnn_experiments_results.pth" --noise-vars "[0.1, 0.5, 1.0, 1.5, 2.0]" --noise-type "layer_variance" --cloud-environment hpc --debug-mode experiment
# python -i dnn_weight_noise.py --experiment-version v36 --experiment-type dnn-load-model --pretrained-model-name "dnn_experiments_results.pth" --noise-vars "[0.1, 0.5, 1.0, 1.5, 2.0]" --noise-type "layer_variance" --cloud-environment hpc --debug-mode experiment
# python -i dnn_weight_noise.py --experiment-version v37 --experiment-type dnn-load-model --pretrained-model-name "dnn_experiments_results.pth" --noise-vars "[0.1, 0.5, 1.0, 1.5, 2.0]" --noise-type "all" --cloud-environment hpc --debug-mode experiment
# python -i dnn_weight_noise.py --experiment-version v38 --experiment-type dnn-load-model --pretrained-model-name "dnn_experiments_results.pth" --noise-vars "[0.1, 0.5, 1.0, 1.5, 2.0]" --noise-type "all" --cloud-environment hpc --debug-mode experiment
# python -i dnn_weight_noise.py --experiment-version v39 --experiment-type dnn-load-model --pretrained-model-name "dnn_experiments_results.pth" --noise-vars "[0.1, 0.5, 1.0, 1.5, 2.0]" --noise-type "all" --cloud-environment hpc --debug-mode experiment
# python -i dnn_weight_noise.py --experiment-version v40 --experiment-type dnn-load-model --pretrained-model-name "dnn_experiments_results.pth" --noise-vars "[0.1, 0.5, 1.0, 1.5, 2.0]" --noise-type "all" --cloud-environment hpc --debug-mode experiment
python -i dnn_weight_noise.py --experiment-version v43 --experiment-type dnn-load-model --pretrained-model-name "dnn_experiments_results.pth" --noise-vars "[0.1, 0.5, 1.0, 1.5, 2.0]" --noise-type "all" --cloud-environment hpc --debug-mode experiment