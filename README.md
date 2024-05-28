```bash
docker run --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:24.04-py3 \
    tritonserver --model-repository=/models
```

docker run --gpus all --rm -it --net host nvcr.io/nvidia/tritonserver:24.04-py3-sdk


 perf_analyzer -m ddnet_jhmdb --async --concurrency-range=10 --collect-metrics -f output.csv --verbose-csv
