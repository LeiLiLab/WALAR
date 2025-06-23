export CUDA_VISIBLE_DEVICES=1

cd /mnt/data1/yifengliu/qe-lr/code

python evaluate_any.py \
  --input_file /mnt/data1/yifengliu/qe-lr/output/IndicMT/Comet-qe-da/eng-maithili.jsonl\
  --output_file /mnt/data1/yifengliu/qe-lr/output/output2.jsonl