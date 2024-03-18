# References:
# 1. Script template: https://github.com/PrincetonUniversity/multi_gpu_training/blob/117b9fc97a21ebc805bfeab0dcce9c1a7b868d9a/02_pytorch_ddp/job.slurm
# 2. NCCL/srun debugging: https://github.com/pytorch/pytorch/issues/76287#issuecomment-1117931318
# 3. Get master node IP address: https://github.com/pytorch/pytorch/issues/25767#issuecomment-1126621943)
script = """#!/bin/bash
####################################################################################################
# Slurm job parameters 
####################################################################################################
#SBATCH --job-name=vdiff         # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of allocated gpus per node
#SBATCH --time=47:59:59          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

####################################################################################################
# Change the working directory to the project location 
####################################################################################################
cd {proj_loc}

####################################################################################################
# Make sure the correct conda env is used, e.g. base
####################################################################################################
export PATH={CONDA_PREFIX}/bin:{PATH}
export LD_LIBRARY_PATH={CONDA_PREFIX}/lib:{LD_LIBRARY_PATH}

####################################################################################################
# Set up necessary environment variables for distributed training 
####################################################################################################
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)
export NNODES=$SLURM_NNODES
export NGPUS=$(nvidia-smi --list-gpus | wc -l)
export WORLD_SIZE=$(($NGPUS * $NNODES))
echo "MASTER_NODE="$MASTER_NODE
echo "MASTER_ADDR="$MASTER_ADDR
echo "NNODES="$NNODES
echo "NGPUS="$NGPUS
echo "WORLD_SIZE="$WORLD_SIZE

####################################################################################################
# Debugging
####################################################################################################
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

####################################################################################################
# srun launch multinode/multiprocess training job
####################################################################################################
export OMP_NUM_THREADS=4
srun --kill-on-bad-exit=1 --wait=60 \\
torchrun \\
  --nproc_per_node=$NGPUS  \\
  --nnodes=$NNODES \\
  --node_rank="$SLURM_PROCID" \\
  --master_addr=$MASTER_ADDR \\
  --master_port=$MASTER_PORT \\
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
  --rdzv_backend=c10d \\
  ./train.py \\
    --config-path ./configs/cifar10_cond.json \\
    --distributed \\
    --use-ddim \\
    --num-save-images 80"""

if __name__ == "__main__":
    import os
    import sys

    if not os.path.exists("./jobs"):
        os.makedirs("./jobs")

    if len(sys.argv) >= 2:
        script = script.format(**os.environ).replace(":\n", "\n")
        command = sys.argv[1]
        if command == "run":
            with open("./jobs/job.slurm", "w") as f:
                f.write(script)
            os.system("sbatch ./jobs/job.slurm")
        elif command == "print":
            print("./jobs/job.slurm")
        else:
            raise NotImplementedError(command)
