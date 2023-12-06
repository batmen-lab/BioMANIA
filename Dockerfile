FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install Python, Node.js, miniconda, dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    bash \
    net-tools \
    iputils-ping \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Conda
RUN curl -sLo /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /miniconda.sh -b -p /miniconda \
    && rm /miniconda.sh
ENV PATH="/miniconda/bin:${PATH}"

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Set the working directory
WORKDIR /app

# Set LIB
ARG LIB

# Copy the backend application
COPY src/requirements.txt    /app/src/requirements.txt
COPY src/LICENSE    /app/src/LICENSE
COPY src/Git2APP /app/src/Git2APP
COPY src/R2APP    /app/src/R2APP
COPY src/configs    /app/src/configs
COPY src/dataloader    /app/src/dataloader
COPY src/deploy    /app/src/deploy
COPY src/gpt    /app/src/gpt
COPY src/inference    /app/src/inference
COPY src/models    /app/src/models
COPY src/plot    /app/src/plot
COPY src/prompt    /app/src/prompt
COPY src/report    /app/src/report
COPY src/retrievers    /app/src/retrievers
#COPY src/tmp    /app/src/tmp
COPY src/data/standard_process/${LIB}/ /app/src/data/standard_process/${LIB}/
COPY src/hugging_models/retriever_model_finetuned/${LIB}/ /app/src/hugging_models/retriever_model_finetuned/${LIB}/
COPY docker_utils/ /app/docker_utils/

# mkdir tmp
RUN mkdir -p /app/src/tmp

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/docker_utils/${LIB}/requirements.txt

# Install dependencies from environment.yml if it exists
RUN if [ -f /app/docker_utils/${LIB}/environment.yml ]; then \
        conda env create -f /app/docker_utils/${LIB}/environment.yml; \
    fi

# Install dependencies from requirements.sh if it exists
RUN if [ -f /app/docker_utils/${LIB}/requirements.sh ]; then \
        chmod +x /app/docker_utils/${LIB}/requirements.sh && \
        /app/docker_utils/${LIB}/requirements.sh; \
    fi

COPY chatbot_ui_biomania/ /app/chatbot_ui_biomania/

# Install node modules and build the front-end
RUN cd /app/chatbot_ui_biomania && npm install && npm run build

# run the start script
RUN chmod +x /app/docker_utils/${LIB}/docker_start_script.sh

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV BACKEND_URL="http://localhost:5000"

# Expose ports
EXPOSE 3000

# Start command
CMD /app/docker_utils/${LIB}/docker_start_script.sh
