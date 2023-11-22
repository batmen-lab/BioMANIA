cd src/
python deploy/inference_dialog_server.py &
backend_pid=$!

cd ../chatbot_ui_biomania/
npm run dev &
frontend_pid=$!

# Keep the script running to keep the processes alive
wait $backend_pid
wait $frontend_pid

