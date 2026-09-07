# LMCache External Fgds Backend

This fgds_backend provides an external fgds backend implementation for LMCache.

## Features
- Implements the `StorageBackendInterface` from LMCache
- Provide a new gpu direct storage backend like 'GDS' for lmcache

## Installation

```bash
python setup install
```

## Usage

1. prepare the vllm test enviroment and get the vllm project for benchmark tool
```bash
git clone https://github.com/vllm-project/vllm.git
```

2. install lmcache
```bash
conda activate vllm-env
pip install lmcache
```

3. start vllm server with LMCache
```bash
cd fgds/python/test
LMCACHE_CONFIG_FILE="./fgds.yaml" LMCACHE_USE_EXPERIMENTAL=True VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=6 vllm serve /data/Qwen-0.6B/ --enable-reasoning --reasoning-parser deepseek_r1 --max-model-len 8192 --port 8022 --gpu-memory-utilization 0.65 -tp 1 --enforce-eager --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

4. vllm benchmark test
```bash
python benchmarks/benchmark_serving.py --model /data/Qwen-0.6B/ --backend vllm --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --host 127.0.0.1 --port 8022 --max-concurrency 64 --num-prompts 128
```