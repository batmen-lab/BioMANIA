source /miniconda/etc/profile.d/conda.sh
conda activate qiime2-amplicon-2023.9

cd /app/src/
python3 deploy/inference_dialog_server.py &
backend_pid=$!

cd /app/chatbot_ui_biomania/
npm run dev &
frontend_pid=$!

# Keep the script running to keep the processes alive
wait $backend_pid
wait $frontend_pid

