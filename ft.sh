#!/bin/bash
#SBATCH --job-name=rahul-405b
#SBATCH --account=root
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --nodelist=mk-xii-08,mk-xii-24
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=22
#SBATCH --time=2000:00:00
#SBATCH --output=/home/rahul/utils/slurm_logs/slurm-%j.out
#SBATCH --error=/home/rahul/utils/slurm_logs/slurm-%j.err


# export LOGLEVEL=INFO
# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=WARN
# export PYTHONFAULTHANDLER=1
# export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
# export CUDA_LAUNCH_BLOCKING=0

# export NCCL_SOCKET_IFNAME="ens"
# export FI_EFA_USE_DEVICE_RDMA=1
# export NCCL_IGNORE_CPU_AFFINITY=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=3600

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


export MASTER_HOSTNAME="mk-xii-08"
export MASTER_ADDR=$(host $MASTER_HOSTNAME | awk '/has address/ { print $4 }')
export MASTER_PORT=29500

# srun bash -c '
#     source $(conda info --base)/etc/profile.d/conda.sh &&
#     conda activate dev &&
#     export NCCL_DEBUG=INFO &&
#     python /home/simarora/code/based/train/run.py experiment=reference/based_2.8b_50b_tok trainer.num_nodes=2
# '

srun torchrun --nnodes 2 --node_rank $SLURM_NODEID --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nproc_per_node 8 /home/rahul/workspace/lolcats/llama_recipes/distill_llama.py --model_config distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 --distill_config distill_alpaca_clean_llama3_1_405b_xent0_mse1000_lr1e-2 --finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_405b --eval_config eval_alpaca_clean --verbose --replicate 0 --seed 0 --lk_zero_init --eval_steps 1 --dataset_chunk_size 1024 --enable_fsdp --fsdp_activation_checkpointing

# /home/simarora/code/based/train/run.py experiment=test_mk12/test trainer.num_nodes=2
# python /home/simarora/code/flash-attention/training/run.py experiment=pile/gpt3s-flash trainer.num_nodes=2
