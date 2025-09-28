# import json
# from typing import Any
# import aiohttp
# import modal
# vllm_image = (
#     modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
#     .uv_pip_install(
#         "vllm==0.10.1.1",
#         "huggingface_hub[hf_transfer]==0.34.4",
#         "flashinfer-python==0.2.8",
#         "torch==2.7.1",
#         "tiktoken>=0.7.0",
#         "blobfile>=3.0.0",
#     )
#     .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
# )
# vllm_image = vllm_image.env({
#     "VLLM_TORCH_COMPILE": "0",  # 彻底关闭 torch.compile
#     "VLLM_USE_CUDA_GRAPH": "0", # 彻底关闭 CUDA Graph 捕获
#     "TORCH_CUDA_ARCH_LIST": "9.0",  # 指定H200的编译架构，避免乱编译
#     "HF_HUB_ENABLE_HF_TRANSFER": "1",
#     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512",
# })
# MODEL_NAME = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
# hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
# vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
# # 暂时禁用V1 API，避免多进程初始化问题
# # vllm_image = vllm_image.env({"VLLM_USE_V1": "1"})
# FAST_BOOT = True
# app = modal.App("example-vllm-inference")
# N_GPU = 1
# MINUTES = 60  # seconds
# VLLM_PORT = 8000
# @app.function(
#     image=vllm_image,
#     gpu=f"A100:{N_GPU}",
#     scaledown_window=60 * MINUTES,  # how long should we stay up with no requests?
#     timeout=1000 * MINUTES,  # how long should we wait for container start?
#     volumes={
#         "/root/.cache/huggingface": hf_cache_vol,
#         "/root/.cache/vllm": vllm_cache_vol,
#     },
# )
# @modal.concurrent(  # how many requests can one replica handle? tune carefully!
#     max_inputs=32
# )
# @modal.web_server(port=VLLM_PORT, startup_timeout=1000 * MINUTES)
# def serve():
#     import subprocess
#     cmd = [
#         "vllm", "serve",
#         MODEL_NAME,
#         "--served-model-name", "kimi",
#         "--host", "0.0.0.0",
#         "--port", str(VLLM_PORT),
#         # 模型加载
#         "--trust-remote-code",
#         "--download-dir", "/root/.cache/huggingface",
#         # 计算/精度
#         "--dtype", "bfloat16",
#         # *** 并行与显存 ***
#         "--tensor-parallel-size", str(N_GPU),      # 8卡并行
#         "--gpu-memory-utilization", "0.95", # 进一步降低显存使用率
        
#         # *** 峰值限制（跟吞吐相关）***
#         "--max-model-len", "32768",         # 进一步降低到32k，减少显存压力
#         "--max-num-seqs", "2",              # 降低到2路并发，减少worker压力
#         "--max-num-batched-tokens", "2048", # 进一步降低预填充峰值
#         # 禁用一些可能导致问题的特性
#         "--disable-log-stats",              # 禁用统计日志
#         "--disable-log-requests",           # 禁用请求日志
#     ]
#     cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
#     print("Launching command:", " ".join(cmd), flush=True)
#     #subprocess.Popen(" ".join(cmd), shell=True)
#     # subprocess.run(cmd, check=True)
#     import os
#     os.execvp(cmd[0], cmd)   # :arrow_left: 关键：替换成前台主进程


# # Installing the CUDA Toolkit on Modal

# This code sample is intended to quickly show how different layers of the CUDA stack are used on Modal.
# For greater detail, see our [guide to using CUDA on Modal](https://modal.com/docs/guide/cuda).

# All Modal Functions with GPUs already have the NVIDIA CUDA drivers,
# NVIDIA System Management Interface, and CUDA Driver API installed.

import modal

app = modal.App("example-install-cuda")


@app.function(gpu="T4")
def nvidia_smi():
    import subprocess

    subprocess.run(["nvidia-smi"], check=True)


# This is enough to install and use many CUDA-dependent libraries, like PyTorch.


@app.function(gpu="T4", image=modal.Image.debian_slim().pip_install("torch"))
def torch_cuda():
    import torch

    print(torch.cuda.get_device_properties("cuda:0"))


# If your application or its dependencies need components of the CUDA toolkit,
# like the `nvcc` compiler driver, installed as system libraries or command-line tools,
# you'll need to install those manually.

# We recommend the official NVIDIA CUDA Docker images from Docker Hub.
# You'll need to add Python 3 and pip with the `add_python` option because the image
# doesn't have these by default.


ctk_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
).entrypoint([])  # removes chatty prints on entry


@app.function(gpu="T4", image=ctk_image)
def nvcc_version():
    import subprocess

    return subprocess.run(["nvcc", "--version"], check=True)


# You can check that all these functions run by invoking this script with `modal run`.


@app.local_entrypoint()
def main():
    nvidia_smi.remote()
    torch_cuda.remote()
    nvcc_version.remote()