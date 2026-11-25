FROM python:3.7-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# 官方仓库原样克隆：保留 encode.cpython-37m-x86_64-linux-gnu.so 与
# libCalculateLeftHands.so，走原版推理链路
RUN git clone --depth 1 https://github.com/Netease-Games-AI-Lab-Guangzhou/PerfectDou.git /app/PerfectDou

# 只保留 perfectdou 原版推理：不需要 torch / rlcard
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY main.py /app/

ENV PYTHONPATH=/app/PerfectDou
ENV PYTHONUNBUFFERED=1
ENV PD_REPO=/app/PerfectDou

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
