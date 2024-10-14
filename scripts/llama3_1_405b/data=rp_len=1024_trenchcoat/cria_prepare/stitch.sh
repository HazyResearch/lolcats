#!/bin/bash
#SBATCH --job-name=llama-405b
#SBATCH --account=root
#SBATCH --partition=batch
#SBATCH --nodes=3
#SBATCH --nodelist=mk-xii-22,mk-xii-24,mk-xii-09
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=19
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

srun  torchrun --nnodes 3 --node_rank $SLURM_NODEID --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nproc_per_node 8 \
    llama_recipes/trenchcoat_lolcat/save_fsdp_to_pt.py \
    --model_config llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_405b/distill_llama_405b_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_405b/finetune_layer_mini_xent1_mse1000 \
    --final_finetune_config llama3_1_405b/finetune_rp_llama_405b_qkvo_e2 \
    --verbose --replicate 0 --seed 0 \
    --layers_per_model 9 --layer_idx 0 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing \
    --load_finetune_checkpoint /path/to/your/sharded/finetuned/checkpoints/ # TODO: Fill in!

# Sample of the entry at load_finetune_checkpoint: 
# /home/simran/code/clean/lolcats/checkpoints/llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01/sharded_layers/finetune-dl-d=llama3_1_70b/distill_rpcontig2048_dcs2048_xent0_mse1000_lr1e-2-m=llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=llama3_1_70b/rp_contig_finetune_llama_70b_qv_hparams_2048-s=0-se=0-re=0-se=0-re=0 
#
# This directory will contain files looking like: __0_0.distcp  __1_0.distcp  __2_0.distcp  __3_0.distcp  __4_0.distcp  __5_0.distcp  __6_0.distcp  __7_0.distcp ... 
# 

# Use the ckpt: You can then take your .pt file and run the 405b demo script in demos/llm_mmlu_eval/ etc!


