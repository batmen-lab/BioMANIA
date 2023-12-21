source /miniconda/etc/profile.d/conda.sh
wget https://data.qiime2.org/distro/amplicon/qiime2-amplicon-2023.9-py38-linux-conda.yml
conda env create -n qiime2-amplicon-2023.9 --file qiime2-amplicon-2023.9-py38-linux-conda.yml
conda activate qiime2-amplicon-2023.9

cd /app/src/
python3 deploy/inference_dialog_server.py &
backend_pid=$!

cd /app/chatbot_ui_biomania/
npm start &
frontend_pid=$!

# Keep the script running to keep the processes alive
wait $backend_pid
wait $frontend_pid

