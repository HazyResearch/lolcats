#!/bin/bash
#SBATCH --job-name=rpqkvo_405b
#SBATCH --account=root
#SBATCH --partition=batch
#SBATCH --nodes=3
#SBATCH --nodelist=mk-turbo-01,mk-turbo-02,mk-turbo-04
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=19
#SBATCH --time=2000:00:00
#SBATCH --output=/home/rahul/code/clean/slurm/lolcats/slurm-%j.out
#SBATCH --error=/home/rahul/code/clean/slurm/lolcats/slurm-%j.err
#SBATCH --ntasks=3                    # Add this line
#SBATCH --ntasks-per-node=1           # Add this line

# Initialize HPC-X toolkit for high-performance computing
. /opt/hpcx/hpcx-init.sh
hpcx_load

export NCCL_IGNORE_CPU_AFFINITY=1  # Ignore CPU affinity settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Enable asynchronous error handling for PyTorch NCCL
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Set CUDA device order to PCI bus ID
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=5  # Enable GPUDirect RDMA for faster GPU-to-GPU communication
export NCCL_P2P_DISABLE=0  # Enable peer-to-peer communication between GPUs
export NCCL_BUFFSIZE=2097152  # Set 2MB buffer size for NCCL operations
export NCCL_IB_HCA=mlx5  # Specify the InfiniBand Host Channel Adapter to use

# Set the master address and port
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_HOSTNAME="mk-turbo-01"
export MASTER_ADDR=$(host $MASTER_HOSTNAME | awk '/has address/ { print $4 }')
export MASTER_PORT=29500
# Set the Python path if needed
export PYTHONPATH=/home/rahul/code/clean/lolcats/

# Run the job using srun
srun --ntasks=3 --ntasks-per-node=1 \
     --gres=gpu:8 \
     --cpus-per-task=19 \
     torchrun --nnodes=3 \
              --nproc_per_node=8 \
              --node_rank=$SLURM_PROCID \
              --rdzv_id=$SLURM_JOB_ID \
              --rdzv_backend=c10d \
              --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
              llama_recipes/stitch_mini_405b_sim2_rp_2048.py \
              --model_config llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 \
              --distill_config llama3_1_405b/distill_rpcontig2048_dcs2048_xent0_mse1000_lr1e-2 \
              --finetune_config llama3_1_405b/rp_contig_finetune_llama_405b_qv_hparams_2048 \
              --final_finetune_config llama3_1_405b/finetune_llama_405b_qkvo_e2_rp_2048 \
              --verbose --replicate 0 --seed 0 \
              --layers_per_model 3 --layer_idx 0 \
              --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing


