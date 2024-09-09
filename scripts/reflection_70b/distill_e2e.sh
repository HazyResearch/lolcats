# #!/bin/bash
# #SBATCH --job-name=reflection-70b
# #SBATCH --account=root
# #SBATCH --partition=batch
# #SBATCH --nodes=2
# #SBATCH --nodelist=mk-xii-13,mk-xii-11
# #SBATCH --gres=gpu:8
# #SBATCH --cpus-per-task=22
# #SBATCH --time=2000:00:00
# #SBATCH --output=/home/simarora/utils/slurm_logs/slurm-%j.out
# #SBATCH --error=/home/simarora/utils/slurm_logs/slurm-%j.err

# # Initialize HPC-X toolkit for high-performance computing
# . /opt/hpcx/hpcx-init.sh
# hpcx_load

# export NCCL_IGNORE_CPU_AFFINITY=1  # Ignore CPU affinity settings
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Enable asynchronous error handling for PyTorch NCCL
# export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Set CUDA device order to PCI bus ID
# export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
# export NCCL_NET_GDR_LEVEL=5  # Enable GPUDirect RDMA for faster GPU-to-GPU communication
# export NCCL_P2P_DISABLE=0  # Enable peer-to-peer communication between GPUs
# export NCCL_BUFFSIZE=2097152  # Set 2MB buffer size for NCCL operations
# export NCCL_IB_HCA=mlx5  # Specify the InfiniBand Host Channel Adapter to use

# export MASTER_HOSTNAME="mk-xii-13"
# export MASTER_ADDR=$(host $MASTER_HOSTNAME | awk '/has address/ { print $4 }')
# export MASTER_PORT=29500
# export PYTHONPATH=/home/simarora/code/lolcats/

# FLAG SCRIPT NAME VS. EXPT (XENT)

# srun  torchrun --nnodes 2 --node_rank $SLURM_NODEID --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/distill_llama.py \
#     --model_config reflection_70b/distill_reflection_model \
#     --distill_config reflection_70b/distill_reflection \
#     --finetune_config reflection_70b/finetune_reflection \
#     --eval_config eval_alpaca_clean \
#     --verbose --replicate 0 --seed 0 \
#     --lk_zero_init \
#     --eval_steps 16 --dataset_chunk_size 1024 \
#     --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing

export PYTHONPATH=/home/simarora/code/lolcats/
torchrun --nnodes 1 --nproc_per_node 8 llama_recipes/distill_llama_finetune.py \
    --model_config reflection_70b/distill_reflection_model \
    --distill_config reflection_70b/distill_reflection \
    --finetune_config reflection_70b/finetune_reflection \
    --eval_config eval_alpaca_clean \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init \
    --eval_steps 100 --dataset_chunk_size 1024 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing

    
