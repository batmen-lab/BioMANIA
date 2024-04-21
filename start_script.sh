ssh -R biomania.serveo.net:80:localhost:3000 -i ~/.ssh/serveo_key serveo.net &
tunnel_pid=$!

python -m src.deploy.inference_dialog_server.py &
backend_pid=$!

cd chatbot_ui_biomania/
npm run dev &
frontend_pid=$!

# Keep the script running to keep the processes alive
wait $backend_pid
wait $frontend_pid
wait $tunnel_pid