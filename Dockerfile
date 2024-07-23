FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

WORKDIR /app

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --upgrade pip

RUN python3.8 -m pip install evaluate==0.4.1 transformers==4.40.1 accelerate==0.30.0 bitsandbytes==0.42.0 torch==2.3.0 peft==0.10.0

CMD ["python3.8"]
