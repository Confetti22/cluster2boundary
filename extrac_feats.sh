#!/bin/bash
#SBATCH --job-name=extrac_feats# Job name
#SBATCH --output=brain_masking_%j.out  # Output file (job ID will be appended)
#SBATCH --error=brain_masking_%j.err   # Error file (job ID will be appended)

#SBATCH --time=10:00:00            # Time limit (here, 1 hour)
#SBATCH --ntasks=1                 # Number of tasks (1 CPU task)
#SBATCH --cpus-per-task=24           # Number of CPU cores to use
#SBATCH --gres=gpu:2
#SBATCH --mem=64G                  # Memory limit (16GB)
#SBATCH --partition=compute         # Partition/queue name (adjust to your cluster)    
#SBATCH --nodelist=c003       # Specify the node you want to run on
# Load any necessary modules
eval "$(conda shell.bash hook)"
conda activate /share/home/shiqiz/.conda/envs/pytorch
# Run the Python script
python /share/home/shiqiz/workspace/cluster2boudnary/generate_feats_map.py