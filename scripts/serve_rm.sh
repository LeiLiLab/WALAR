eval "$(/mnt/gemini/dat1/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/data1/yifengliu/miniconda3/bin/activate qe-rl

cd /mnt/gemini/data1/yifengliu/qe-lr/openrlhf

src="en"
tgt="zh"

export CUDA_VISIBLE_DEVICES=0
python -m openrlhf.cli.serve_rm \
    --model_name  metricX\
    --port 5000 \
    --max_len 1536 \
    --src $src \
    --tgt $tgt \
    --batch_size 8 &

echo "QE MetricX serves successfully!"

export CUDA_VISIBLE_DEVICES=4
python -m openrlhf.cli.serve_rm \
    --model_name  metricX-ref\
    --port 4000 \
    --max_len 1536 \
    --src $src \
    --tgt $tgt \
    --batch_size 8 &

echo "Ref MetricX serves successfully!"

export CUDA_VISIBLE_DEVICES=2
python -m openrlhf.cli.serve_rm \
    --model_name Comet22\
    --port 7000 \
    --max_len 1536 \
    --src $src \
    --tgt $tgt \
    --batch_size 16 &

echo "COMET22 serves successfully!"


export CUDA_VISIBLE_DEVICES=3
python -m openrlhf.cli.serve_rm \
    --model_name XComet\
    --port 8000 \
    --max_len 1536 \
    --src $src \
    --tgt $tgt \
    --batch_size 16 &

echo "XComet serves successfully!"

wait