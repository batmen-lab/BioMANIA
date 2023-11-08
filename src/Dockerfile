# 使用支持CUDA的基础镜像
FROM nvidia/cuda:11.0-runtime

# 安装Python和pip
RUN apt-get update && apt-get install -y python3 python3-pip

# 创建并设置工作目录
WORKDIR /app

# 复制requirements.txt文件，并安装Python依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制后端源代码
COPY . .

# 设置环境变量
ENV LIB=scanpy
ENV CUDA_VISIBLE_DEVICES=0

# 暴露后端服务端口
EXPOSE 5000

# 运行Flask应用
CMD ["sh", "-c", "LIB=${LIB} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 deploy/inference_dialog_server.py --retrieval_model_path ./hugging_models/retriever_model_finetuned/${LIB}/assigned --top_k 3"]
