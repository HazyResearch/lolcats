#!/bin/bash
#SBATCH --job-name=llama-405b
#SBATCH --account=root
#SBATCH --partition=batch
#SBATCH --nodes=3
#SBATCH --nodelist=mk-xii-02,mk-xii-24,mk-xii-08
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=22
#SBATCH --time=2000:00:00
#SBATCH --output=/home/simarora/utils/slurm_logs/slurm-%j.out
#SBATCH --error=/home/simarora/utils/slurm_logs/slurm-%j.err

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

export MASTER_HOSTNAME="mk-xii-24"
export MASTER_ADDR=$(host $MASTER_HOSTNAME | awk '/has address/ { print $4 }')
export MASTER_PORT=29500
export PYTHONPATH=/home/simarora/code/lolcats/


srun  torchrun --nnodes 3 --node_rank $SLURM_NODEID --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/distill_llama.py \
    --model_config llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_405b/distill_llama_405b_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_405b/finetune_llama_405b \
    --eval_config eval_alpaca_clean \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init --eval_steps 16 --dataset_chunk_size 1024 \
    --enable_fsdp --fsdp_activation_checkpointing


# srun  torchrun --nnodes 3 --node_rank $SLURM_NODEID --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/distill_llama.py --model_config distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 --distill_config distill_alpaca_clean_llama3_1_405b_xent0_mse1000_lr1e-2 --finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_405b --eval_config eval_alpaca_clean --verbose --replicate 0 --seed 0 --lk_zero_init --eval_steps 1 --dataset_chunk_size 512 --enable_fsdp --fsdp_activation_checkpointing

# srun  torchrun --nnodes 3 --node_rank $SLURM_NODEID --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/distill_llama_finetune.py --model_config distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 --distill_config distill_alpaca_clean_llama3_1_405b_xent0_mse1000_lr1e-2 --finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_405b --eval_config eval_alpaca_clean --verbose --replicate 0 --seed 0 --lk_zero_init --eval_steps 1 --dataset_chunk_size 512 --enable_fsdp --fsdp_activation_checkpointing








# Ablation with xent1
# srun  torchrun --nnodes 3 --node_rank $SLURM_NODEID --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/distill_llama.py --model_config distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 --distill_config distill_alpaca_clean_llama3_1_405b_xent1_mse1000_lr1e-2 --finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_405b --eval_config eval_alpaca_clean --verbose --replicate 0 --seed 0 --lk_zero_init --eval_steps 100 --dataset_chunk_size 512 --enable_fsdp --fsdp_activation_checkpointing




