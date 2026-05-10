#!/bin/bash -l
#SBATCH --job-name=gdr-esdg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=OD-210257
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=120:00:00

set -euo pipefail

echo "Job $SLURM_JOB_ID started at $(date)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_ALLOW_TF32=0
export CUBLAS_ALLOW_TF32=0

source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

chmod +x ./run_esdg.sh

export OUTPUT_DIR="./output_esdg_h100"

set +e

./run_esdg.sh

set -e

echo "Python script finished. Checking checkpoints for status..."
echo "Job finished at $(date)"
