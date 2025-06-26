#!/bin/bash

# Default values
declare -A model_path

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

model_path["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen2.5-0.5B-Instruct"
model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/Hybrid-Qwen2.5-0.5B-en-zh-1M-bsz128/global_step180_hf"

MODEL_NAME="Qwen"
MODEL_PATH=${model_path[$MODEL_NAME]}
# zho_simpl, zho_trad, swh, tam
LANG_PAIR="eng-zho_simpl"
INPUT_DIR="/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest"
OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores"
PORT=1234


server=True

if [ "$server" = True ]; then
    python3 -m sglang.launch_server --model-path ${MODEL_PATH} --host 0.0.0.0 --port ${PORT}
else
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    # Generate output filename
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}/flores_${LANG_PAIR}.txt"

    cd /mnt/gemini/data1/yifengliu/qe-lr
    # Run the evaluation

    python evaluate/flores_sglang.py \
        --model_name_or_path "$MODEL_PATH"\
        --data_dir "$INPUT_DIR"\
        --lang_pair "$LANG_PAIR" \
        --output_file "$OUTPUT_FILE"\
        --port ${PORT}
fi