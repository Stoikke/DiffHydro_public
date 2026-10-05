#!/usr/bin/env bash
#SBATCH --job-name=stromgren_validation
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpu-bind=none
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=pbs/%x_%j.out
#SBATCH --error=pbs/%x_%j.err
#SBATCH --mem-per-cpu=30GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=adrien.bernard@etu.u-bordeaux.fr
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1

source ~/.bashrc
cd /home/adrien/DiffHydro_public

eval "$(conda shell.bash hook)"
conda activate astro-env
export PYTHONPATH="/home/adrien/DiffHydro_public:${PYTHONPATH:-}"
export NCCL_LIB_DIR="/home/adrien/miniconda3/envs/astro-env/lib/python3.11/site-packages/nvidia/nccl/lib"
export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"

mkdir -p pbs

format_seconds() {
    local total_seconds="$1"
    local days=$((total_seconds / 86400))
    local hours=$(((total_seconds % 86400) / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    if (( days > 0 )); then
        printf "%dd %02dh %02dm %02ds" \
            "$days" "$hours" "$minutes" "$seconds"
    else
        printf "%02dh %02dm %02ds" \
            "$hours" "$minutes" "$seconds"
    fi
}

echo "============================================================"
echo "Job ID        : ${SLURM_JOB_ID:-unknown}"
echo "Node          : $(hostname)"
echo "Start date    : $(date '+%F %T %Z')"
echo "GPUs visible  : ${CUDA_VISIBLE_DEVICES}"
echo "Working dir   : $(pwd)"
echo "============================================================"

job_start_epoch=$(date +%s)

echo
echo "============================================================"
echo "START 1/2 — Stromgren validation"
echo "============================================================"

stromgren_start_epoch=$(date +%s)

srun --ntasks=1 --exclusive \
    env N=150 python examples/RT/serveur/stromgren_validation_simple_node.py

stromgren_exit_code=$?
stromgren_end_epoch=$(date +%s)
stromgren_elapsed=$((stromgren_end_epoch - stromgren_start_epoch))

echo "END 1/2 — Stromgren validation"
echo "Exit code     : ${stromgren_exit_code}"
echo "Elapsed       : $(format_seconds "${stromgren_elapsed}")"
echo "Elapsed raw   : ${stromgren_elapsed} s"
echo "Finished at   : $(date '+%F %T %Z')"
echo "============================================================"

if (( stromgren_exit_code != 0 )); then
    echo "ERROR: Stromgren validation failed; second simulation is not started."
    exit "${stromgren_exit_code}"
fi

echo
echo "============================================================"
echo "START 2/2 — RT_run_chat_v5"
echo "============================================================"

rt_start_epoch=$(date +%s)

srun --ntasks=1 --exclusive \
    env python examples/RT/serveur/RT_run_chat_v5.py

rt_exit_code=$?
rt_end_epoch=$(date +%s)
rt_elapsed=$((rt_end_epoch - rt_start_epoch))

echo "END 2/2 — RT_run_chat_v5"
echo "Exit code     : ${rt_exit_code}"
echo "Elapsed       : $(format_seconds "${rt_elapsed}")"
echo "Elapsed raw   : ${rt_elapsed} s"
echo "Finished at   : $(date '+%F %T %Z')"
echo "============================================================"

job_end_epoch=$(date +%s)
job_elapsed=$((job_end_epoch - job_start_epoch))

echo
echo "============================================================"
echo "RUNTIME SUMMARY"
echo "============================================================"
printf "%-32s %s\n" \
    "Stromgren validation:" "$(format_seconds "${stromgren_elapsed}")"
printf "%-32s %s\n" \
    "RT_run_chat_v5:" "$(format_seconds "${rt_elapsed}")"
printf "%-32s %s\n" \
    "Total bash job runtime:" "$(format_seconds "${job_elapsed}")"
echo
printf "%-32s %s s\n" "Stromgren raw:" "${stromgren_elapsed}"
printf "%-32s %s s\n" "RT_run_chat_v5 raw:" "${rt_elapsed}"
printf "%-32s %s s\n" "Total raw:" "${job_elapsed}"
echo "End date      : $(date '+%F %T %Z')"
echo "============================================================"

exit "${rt_exit_code}"