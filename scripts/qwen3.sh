#!/bin/bash
# Default values

declare -A model_path
export CUDA_VISIBLE_DEVICES=1,2,3,4
eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

model_path["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen3-32B"

# "/mnt/gemini/data1/yifengliu/model/Qwen3-235B-A22B-GPTQ-Int4"

# /mnt/gemini/data1/yifengliu/checkpoints/Qwen2.5-0.5B-En-Zh-1M-bsz128/global_step140_hf
# /mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-3B-en-zh-1M-bsz128/global_step120_hf
# /mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-3B-en-ru-1M-bsz128/global_step120_hf
# /mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-3B-en-zh-1M-bsz128/global_step120_hf

MODEL_NAME="Qwen"
MAX_TOKENS=2048
EVAL_TYPE="mqm"
MODEL_PATH=${model_path[$MODEL_NAME]}
# zho_simpl, zho_trad, swh, tam, fra, rus
# spa(Spanish), deu(German)， heb(Hebrew)
# ben(Bengali), hin(Hindi)
# jpn(Japanese)
# tgl(fillipino Tagalog)
# fin(Finnish)
# ara(Arabic)
# tur(Turkish)
# LANG_PAIR="zho_simpl-deu"
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
INPUT_FILE="/mnt/gemini/data1/yifengliu/data/IndicMT/collated/punjabi2.jsonl"
server=True
# 1234

if [ $MODEL_NAME == "Qwen" ]; then
    relative_path=${MODEL_PATH#*/model/}
else
    relative_path=${MODEL_PATH#*/checkpoints/}
fi
OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/temp"


# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
    # Generate output filename

cd /mnt/gemini/data1/yifengliu/qe-lr
# Run the evaluation
echo "evaluating ${LANG_PAIR} with model ${MODEL_NAME} at ${INPUT_FILE}"
python code/qwen3.py \
    --model_name_or_path "$MODEL_PATH"\
    --input_file "$INPUT_FILE" \
    --max_tokens "$MAX_TOKENS" \
    --eval_type "$EVAL_TYPE" \
    --tensor_parallel_size  $num_gpus \
    --output_dir "$OUTPUT_DIR" \
