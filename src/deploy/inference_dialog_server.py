"""
Author: Zhengyuan Dong
Email: zydong122@gmail.com
Description: This script contains functions to run the BioMANIA model using Flask.
"""
# Flask
from flask import Flask, Response, stream_with_context, request
from flask_socketio import SocketIO
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
# standard lib
import json, signal, time, base64, requests, importlib, inspect, ast, os, random, io, sys, pickle, shutil, subprocess
import traceback
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime
# Computational
from typing import Any
from ..deploy.utils import basic_types, generate_api_calling, download_file_from_google_drive, download_data, save_decoded_file, correct_bool_values, convert_bool_values, infer, dataframe_to_markdown, convert_image_to_base64, change_format
from ..configs.model_config import *
from loguru import logger
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
os.makedirs(f"./logs", exist_ok=True)
logger.remove()
logger.add(f"./logs/BioMANIA_log_{timestamp}.log", rotation="500 MB", retention="7 days", level="INFO")
logger.info("Loguru initialized successfully.")
print("Logging setup complete.")

# device
import torch
if torch.cuda.is_available():
    gpu_index = 2
    torch.cuda.set_device(gpu_index)
    device = torch.device('cuda')
    print("Current GPU Index: {}", torch.cuda.current_device())
else:
    device = torch.device('cpu')
    print("Current Using CPU")
# 
import concurrent.futures
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
# inference pipeline
if not os.path.exists('tmp'):
    os.mkdir('tmp')

from ..deploy.model import Model
import threading
model = Model(logger,device)
should_stop = threading.Event()

def log_interaction(data, method):
    with open('log_file.txt', 'a') as log_file:
        log_file.write(f"{method} - {json.dumps(data)}\n")

def save_interaction_data(data):
    with open('interaction_data.json', 'a') as f:
        json.dump(data, f)
        f.write('\n')

@app.route('/api/stop_generation', methods=['POST'])
def stop_generation():
    global should_stop
    should_stop.set()
    return Response(json.dumps({"status": "stopped"}), status=200, mimetype='application/json')

@app.route('/stream', methods=['GET', 'POST'])
@cross_origin()
def stream():
    data = json.loads(request.data)
    log_interaction(data, request.method) # record received data immediately
    #save_interaction_data(data)
    logger.info('='*30)
    logger.info('get data:')
    for key, value in data.items():
        if key not in ['files']:
            logger.info('{}: {}', key, value)
    logger.info('='*30)
    # makeup for top_k
    if 'top_k' not in data:
        data['top_k'] = 1
    #user_input, top_k, Lib, conversation_started = data["text"], data["top_k"], data["Lib"], data["conversation_started"]
    raw_files = data["files"]
    #session_id = data['session_id']
    try:
        new_lib_doc_url = data["new_lib_doc_url"]
        new_lib_github_url = data["new_lib_github_url"]
        api_html = data['api_html']
        lib_alias = data['lib_alias']
    except:
        # TODO: this is a bug, the default value will also be ""
        new_lib_doc_url = ""
        new_lib_github_url = ""
        api_html = ""
        lib_alias = ""
    # process uploaded files
    if len(raw_files)>0:
        logger.info('length of files: {}',len(raw_files))
        for i in range(len(raw_files)):
            try:
                logger.info(str(raw_files[i]['data'].split(",")[0]))
                logger.info(str(raw_files[i]['filename']))
            except:
                pass
        files = [save_decoded_file(raw_file) for raw_file in raw_files]
        files = [i for i in files if i] # remove None
    else:
        files = []
    global model
    def generate(model):
        global should_stop
        logger.info("Called generate")
        if model.inuse:
            return Response(json.dumps({
                "method_name": "error",
                "error": "Model in use"
            }), status=409, mimetype='application/json')
            return
        model.inuse = True
        """if lib_alias:
            logger.info(lib_alias)
            logger.info('new_lib_doc_url is not none, start installing lib!')
            model.install_lib(data["Lib"], lib_alias, api_html, new_lib_github_url, new_lib_doc_url)"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            logger.info('start running pipeline!')
            print('data:', data)
            future = executor.submit(model.run_pipeline, data["text"], data["Lib"], data["top_k"], files, data["conversation_started"], data['session_id'], data["mode"])
            # keep waiting for the queue to be empty
            while True:
                if should_stop.is_set():
                    should_stop.clear()
                    model.inuse = False
                    executor.shutdown(wait=False)
                    yield json.dumps({"method_name": "generation_stopped"})
                    return
                time.sleep(0.1)
                if model.queue.empty():
                    if future.done():
                        logger.info("Finished with future")
                        break
                    time.sleep(0.01)
                    continue
                else:
                    obj = model.queue.get()
                    log_interaction(obj, 'QUEUE')
                if obj["method_name"] == "unknown": continue
                if obj["method_name"] == "on_request_end":
                    yield json.dumps(obj)
                    break
                try:
                    yield json.dumps(obj) + "\n"
                except Exception as e:
                    e = traceback.format_exc()
                    print('Error:', e)
                    model.inuse = False
            try:
                future.result()
            except Exception as e:
                e = traceback.format_exc()
                print('Error:', e)
                model.inuse = False
        model.inuse = False
        return
    return Response(stream_with_context(generate(model)))

@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    data = json.loads(request.data)
    api_key = data.get('apiKey')
    os.environ["OPENAI_API_KEY"] = api_key
    return Response(json.dumps({"status": "success"}), status=200, mimetype='application/json')

"""GITHUB_CLIENT_ID = os.environ.get('GITHUB_CLIENT_ID')
GITHUB_CLIENT_SECRET = os.environ.get('GITHUB_CLIENT_SECRET')
"""

if __name__ == '__main__':
    def handle_keyboard_interrupt(signal, frame):
        global model
        exit(0)
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    app.run(use_reloader=False, host="0.0.0.0", debug=True, port=5000)
