#!/bin/bash

set -x

template_type=base
model_path=/mnt/gemini/data1/yifengliu/model/Qwen2.5-1.5B #set your model path
qe_model_path=/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt #set your metric ckpt

train_file_path="/mnt/gemini/data1/yifengliu/qe-lr/data/train/base-en-zh_simpl.parquet"
val_file_path="/mnt/gemini/data1/yifengliu/qe-lr/data/val/base-en-zh_simpl.parquet"

src="en"
tgt="zh"


### Step 2: MT-R1-Zero Training
export WANDB_API_KEY=5bebcc325992863eb55622d9ad2e7c85c95a1f15 # set your wandb api key

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
datetime=$(date +"%Y%m%d%H%M%S")
echo $datetime

train_batch_size=8
rollout_num=2
comet_free_rm=False 
reward_metric=Model # []'Model', 'BLEU', 'Merge'] If use reward_metric=BLEU, set comet_rm and comet_free_rm False
exp_name=${src}-${tgt}-bs@${train_batch_size}_n@${rollout_num}_qe_rm@${comet_free_rm}_reward_metric@${reward_metric}_@${datetime}

qe_model_enable=True
micro_batch_size=64

cd /mnt/gemini/data1/yifengliu/qe-lr

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=/mnt/gemini/data1/yifengliu/qe-lr/verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_file_path} \
    data.val_files=${val_file_path} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    qe_model.enable=${qe_model_enable} \
    qe_model.use_valid=${qe_model_enable} \
    qe_model.ckpt_path=${qe_model_path} \
    qe_model.forward_micro_batch_size=${micro_batch_size}\
    qe_model.src_lang=${src} \
    qe_model.tgt_lang=${tgt} \
    algorithm.reward_type='continuous' \
    algorithm.use_kl_in_reward=True \
    algorithm.reward_metric=${reward_metric} \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.check_think=True \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=/mnt/gemini/data1/yifengliu/qe-lr/log/rollout_data \
    trainer.validation_data_dir=/mnt/gemini/data1/yifengliu/qe-lr/log/validation_data \
    trainer.logger=['wandb'] \
    trainer.project_name='QE-RL' \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.default_local_dir=${exp_name} \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=1000 \
    trainer.test_freq=200 \
    trainer.total_epochs=1 $@ 2>&1 | tee grpo.log