#!/bin/bash
#SBATCH --job-name=llama-405b
#SBATCH --account=root
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --nodelist=mk-turbo-02,mk-turbo-04  
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=22
#SBATCH --time=2000:00:00
#SBATCH --output=/home/rahul/code/clean/slurm/slurm-%j.out 
#SBATCH --error=/home/rahul/code/clean/slurm/slurm-%j.err

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

export MASTER_HOSTNAME="mk-turbo-04"                      # # TODO change to your nodenames
export MASTER_ADDR=$(host $MASTER_HOSTNAME | awk '/has address/ { print $4 }')
export MASTER_PORT=29500

export PYTHONPATH=/home/rahul/code/clean/lolcats/         # TODO change to your folder

# Save the model outputs
srun  torchrun --nnodes 2 --node_rank $SLURM_NODEID --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nproc_per_node 8 /home/rahul/code/clean/lolcats/llama_recipes/evals/eval_mmlu.py \
    --model_config llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_405b/distill_llama_405b_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_405b/finetune_llama_405b_qkvo_e2 \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init \
    --eval_steps 100 --dataset_chunk_size 1024 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing \
    --finetune_checkpoint_path /home/rahul/code/clean/lolcats/ckpt_lora-dl-d=distill_rpcontig2048_dcs2048_xent0_mse1000_lr1e-2-m=distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=rp_contig_finetune_llama_405b_qv_hparams_2048-s=0-se=0-re=0-ef=finetune_llama_405b_qkvo_e2_rp_2048-ft_lora=0-se=0-re=0.pt

