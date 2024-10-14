#!/bin/bash
#SBATCH --job-name=llama-405b
#SBATCH --partition=sixhour
#SBATCH --nodes=2
#SBATCH --nodelist=mk-xii-05,mk-xii-06   # TODO: set to your nodenames
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=19
#SBATCH --time=5:59:00
#SBATCH --output=/home/simarora/utils/slurm_logs/slurm-%j.out    # TODO: make your own directory
#SBATCH --error=/home/simarora/utils/slurm_logs/slurm-%j.err
#SBATCH --ntasks=2                    # Add this line
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

export MASTER_HOSTNAME="mk-xii-05"                      # # TODO change to your nodenames
export MASTER_ADDR=$(host $MASTER_HOSTNAME | awk '/has address/ { print $4 }')
export MASTER_PORT=29500

export PYTHONPATH=/home/simarora/code/lolcats/        # TODO change to your folder

# Save the model outputs
srun  torchrun --nnodes 2 --node_rank $SLURM_NODEID --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nproc_per_node 8 \
    demos/llm_mmlu_eval/eval_mmlu.py \
    --model_config llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_405b/rp_distill_llama_405b_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_405b/finetune_rp_llama_405b_qkvo_e2 \
    --verbose --replicate 0 --seed 0 --lk_zero_init \
    --eval_steps 100 --dataset_chunk_size 1024 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing \
    --tag hf_405b_mmlu \
    --finetune_checkpoint_path hazyresearch/lolcats-llama-3.1-405b


# Alternatively, you can run with your own locally trained paths by passing in the the checkpoint_path like follows:
# --finetune_checkpoint_path /home/simarora/code/lolcats/checkpoints/ckpt_lora-dl-d=rp_distill_llama_405b_xent1_mse1000_lr1e-2-m=distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=rp_finetune_llama_40b_qv_hparams-s=0-se=0-re=0-ef=finetune_llama_405b_qkvo_e2_rp-ft_lora=0-se=0-re=0-s=3550.pt



