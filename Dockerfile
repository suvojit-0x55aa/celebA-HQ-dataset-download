FROM continuumio/miniconda3

RUN apt-get update \
    && apt-get install -y -q p7zip-full \
    && rm -rf /var/lib/apt/lists/*

RUN conda install jpeg=8d tqdm requests pillow==3.1.1 urllib3 numpy cryptography scipy

RUN pip install opencv-python==3.4.0.12 cryptography==2.1.4

COPY . /workspace

WORKDIR /data

CMD ["sh", "/workspace/create_celebA-HQ.sh", "/data"]
