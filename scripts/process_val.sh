src="en"
tgt="zh_simpl"
data_source="flores101"

model_path=/mnt/gemini/data1/yifengliu/model/Qwen2.5-3B-Instruct #set your model path
template_type=base
output_file_path=/mnt/gemini/data1/yifengliu/qe-lr/data/val/${template_type}-${src}-${tgt}.parquet


eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl


python3 /mnt/gemini/data1/yifengliu/qe-lr/code/process_val.py \
    --src ${src} \
    --tgt ${tgt} \
    --data_source ${data_source} \
    --tokenizer_path ${model_path} \
    --template_type ${template_type} \
    --output_file ${output_file_path} 

