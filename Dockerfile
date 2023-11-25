FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install Python and Node.js dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    bash \
    net-tools \
    iputils-ping \
    curl

RUN curl -fsSL https://deb.nodesource.com/setup_19.x | bash - && \
    apt-get install -y nodejs

# Set the working directory
WORKDIR /app

# Copy the backend application
COPY src/ /app/src/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/src/requirements.txt

# Copy the front-end application
COPY chatbot_ui_biomania/ /app/chatbot_ui_biomania/

# Install node modules and build the front-end
RUN cd /app/chatbot_ui_biomania && npm install && npm run build

# Copy the start script
COPY start_script.sh /app/
RUN chmod +x /app/start_script.sh

# Set environment variables
ENV LIB=scanpy
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV BACKEND_URL="http://localhost:5000"

# Expose ports
EXPOSE 3000

# Start command
CMD ["/app/start_script.sh"]
