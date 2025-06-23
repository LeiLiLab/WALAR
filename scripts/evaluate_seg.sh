export CUDA_VISIBLE_DEVICES=1

cd /mnt/data1/yifengliu/qe-lr/code


python evaluate_seg.py \
  --input_file /mnt/data1/yifengliu/qe-lr/output/afriMTE/metricX/afriMTE.jsonl\
  --output_file /mnt/data1/yifengliu/qe-lr/output/output2.jsonl
