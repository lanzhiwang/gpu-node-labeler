FROM python:3.12.15

WORKDIR /app

# 安装必要的依赖 (nvidia-ml-py 对应最新 NVML bindings)
RUN pip install --no-cache-dir \
    nvidia-ml-py==12.560.30 \
    kubernetes==30.1.0

COPY gpu_node_labeler.py /app/gpu_node_labeler.py
RUN chmod +x /app/gpu_node_labeler.py

CMD ["python", "-u", "/app/gpu_node_labeler.py"]
