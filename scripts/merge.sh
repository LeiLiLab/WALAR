eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

cd /mnt/gemini/data1/yifengliu/qe-lr

LOCAL_DIR="/mnt/gemini/data1/yifengliu/checkpoints/en-zh-bs@32_n@8_metricX-2w/global_step_200/actor"
TARGET_DIR="/mnt/gemini/data1/yifengliu/checkpoints/en-zh-bs@32_n@8_metricX-2w/global_step_200/hf_model"

export PYTHONPATH=$(pwd):$PYTHONPATH

python code/model_merger.py merge \
    --backend fsdp \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR
