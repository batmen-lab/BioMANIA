git clone https://github.com/aertslab/scenicplus
cd scenicplus
pip install -e .
cd ..

cd /app/
python3 -m src.deploy.inference_dialog_server &
backend_pid=$!

cd /app/chatbot_ui_biomania/
npm start &
frontend_pid=$!

# Keep the script running to keep the processes alive
wait $backend_pid
wait $frontend_pid

