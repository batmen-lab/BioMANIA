git clone https://github.com/aertslab/scenicplus
cd scenicplus
pip install -e .
cd ..

cd /app/src/
python3 deploy/inference_dialog_server.py &
backend_pid=$!

cd /app/chatbot_ui_biomania/
npm run dev &
frontend_pid=$!

# Keep the script running to keep the processes alive
wait $backend_pid
wait $frontend_pid

