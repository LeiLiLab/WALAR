# ray start --head --node-ip-address 0.0.0.0 --num-gpus 4


export CUDA_VISIBLE_DEVICES=6,7

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

export DS_SKIP_CUDA_CHECK=1
cd /mnt/gemini/data1/yifengliu/qe-lr

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/mnt/gemini/data1/yifengliu/qe-lr/openrlhf", "env_vars": { "PYTHONPATH": "/mnt/gemini/data1/yifengliu/qe-lr"}}' \
    -- python -m cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --ref_reward_offload \
    --pretrain /mnt/gemini/data1/yifengliu/model/Qwen2.5-3B-Instruct \
    --remote_rm_url http://localhost:5000/get_reward \
    --save_path /mnt/gemini/data1/yifengliu/qe-lr/checkpoint/Qwen2.5-3B-Instruct \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 128 \
    --n_samples_per_prompt 1\
    --max_samples 100000 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --packing_samples \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --use_kl_loss\
    --init_kl_coef 0.01 \
    --advantage_estimator group_norm \
    --prompt_data /mnt/gemini/data1/yifengliu/qe-lr/data/test_base_en.jsonl \
    --input_key input_key \
    --apply_chat_template True\
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 10\
    --use_wandb {wandb_token}
