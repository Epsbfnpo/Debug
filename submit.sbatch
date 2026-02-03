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
#SBATCH --time=2:00:00

set -euo pipefail

echo "Job $SLURM_JOB_ID started at $(date)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

chmod +x ./run_esdg.sh

export OUTPUT_DIR="./output_esdg_h100"

set +e

./run_esdg.sh

set -e

echo "Python script finished. Checking checkpoints for status..."

CHECK_DIR="${OUTPUT_DIR}/GDRNet_ESDG_MESSIDOR"

FINAL_PTH="${CHECK_DIR}/final_model.pth"
LATEST_PTH="${CHECK_DIR}/latest_model.pth"

echo "Checking directory: $CHECK_DIR"

if [ -f "$FINAL_PTH" ]; then
    echo "✅ Found final_model.pth. Training completed successfully."
    exit 0
elif [ -f "$LATEST_PTH" ]; then
    echo "🔄 Found latest_model.pth but NO final_model.pth."
    echo "⏳ This indicates training timed out or was interrupted. Resubmitting..."
    sbatch submit.sbatch
    exit 0
else
    echo "❌ Neither final nor latest checkpoint found."
    echo "   This implies the job failed before the first epoch or path is wrong."
    exit 1
fi

echo "Job finished at $(date)"