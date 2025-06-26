#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB                    # all mem avail
#SBATCH --gpus=6
#SBATCH --partition=gemini
#SBATCH --account=yifengliu
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --output=/mnt/gemini/data1/yifengliu/logs/grpo-stdout2.txt
#SBATCH --error=/mnt/gemini/data1/yifengliu/logs/grpo-stderr2.txt
#SBATCH --overcommit

# project settings
OPENRLHF_PATH="/mnt/gemini/data1/yifengliu/qe-lr/openrlhf"
MOUNT="$OPENRLHF_PATH:/openrlhf,$HOME/.cache:/root/.cache"
RAY_VERSION=2.12.0

JOBLOG="/mnt/gemini/data1/yifengliu/logs/train_grpo_qwen_hybrid-$SLURM_JOB_ID.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# launch ray daemon
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
echo "Nodes: $nodes" &>> ${JOBLOG}
nodes_array=( $nodes )
node_1=${nodes_array[0]}
ip=$node_1

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"  &>> ${JOBLOG}

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

echo "STARTING HEAD at $node_1"  &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 -w "$node_1" bash -c \
"ray start --head --node-ip-address=$ip --port=$port --block" &>> ${JOBLOG} &
sleep 10s

worker_num=$((SLURM_JOB_NUM_NODES)) #number of nodes other than the head node
for ((i = 1; i < worker_num; i++)); do
node_i=${nodes_array[$i]}
echo "STARTING WORKER $i at $node_i"  &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 -w "$node_i" bash -c \
    "ray start --address $ip_head --block" &>> ${JOBLOG} &
sleep 1s;
done

sleep 30s
echo "Ray cluster started successfully!" &>> ${JOBLOG}

echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

export DS_SKIP_CUDA_CHECK=1
wandb_token=5bebcc325992863eb55622d9ad2e7c85c95a1f15

# ===== submit ray job =====
# Job start
srun --exact --nodes=1 --ntasks=1 -w "$node_1" bash -c \
"ray job submit --address=http://localhost:${port} \
    --runtime-env-json='{"working_dir": "/mnt/gemini/data1/yifengliu/qe-lr/openrlhf"}' \
    -- python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --ref_reward_offload \
    --pretrain /mnt/gemini/data1/yifengliu/model/Qwen2.5-0.5B-Instruct \
    --remote_rm_url http://localhost:5000/get_reward \
    --remote_rm_url2 http://localhost:4000/get_reward \
    --remote_comet_url http://localhost:7000/get_reward \
    --remote_metric_reference_url http://localhost:4000/get_reward \
    --micro_train_batch_size 32 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 32 \
    --rollout_batch_size 128 \
    --n_samples_per_prompt 8 \
    --max_samples 1000000 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 2048 \
    --packing_samples \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --use_kl_loss \
    --init_kl_coef 0.01 \
    --kl_estimator k3 \
    --advantage_estimator group_norm \
    --prompt_data /mnt/gemini/data1/yifengliu/qe-lr/data/train/base_${src}-${tgt}-1m.jsonl \
    --src ${src} \
    --tgt ${tgt} \
    --eval_dir "/mnt/gemini/data1/yifengliu/data/flores101_dataset/dev" \
    --eval_temperature 0.0 \
    --eval_steps 10000 \
    --eval_n_samples_per_prompt 1\
    --input_key input_key \
    --apply_chat_template \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --temperature 1 \
    --save_steps 10 \
    --save_path /mnt/gemini/data1/yifengliu/checkpoints/final/Hybrid-Qwen2.5-0.5B-${src}-${tgt}-1M-bsz128 \
    --ckpt_path /mnt/gemini/data1/yifengliu/checkpoints/Hybrid-Qwen2.5-0.5B-${src}-${tgt}-1M-bsz128 \
    --load_checkpoint \
    --save_hf_ckpt \
    --use_wandb ${wandb_token} \
    --wandb_run_name "Hybrid-Qwen2.5-0.5B-${src}-${tgt}-1M-bsz128"" &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}
