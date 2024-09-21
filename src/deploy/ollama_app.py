"""
Author: Zhengyuan Dong
Email: zydong122@gmail.com
Description: The code contains functions to wrap the BioMANIA model interaction into the ollama supported format.
"""
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ..deploy.ollama_demo import initialize_model, process_input 
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Initialize the model once when the server starts
model = initialize_model()
library = "scanpy"  # Set default library or fetch dynamically based on your requirements

def generate_stream(user_input, session_id, conversation_started):
    responses = process_input(model, user_input, library, session_id, conversation_started)
    for response in responses:
        chunk = {
            "model": "biomania",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "message": {
                "role": "assistant",
                "content": response
            },
            "done": False
        }
        yield f"{json.dumps(chunk)}\n"
    # Ensure the last response indicates completion
    final_chunk = {
        "model": "biomania",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "message": {
            "role": "assistant",
            "content": ""
        },
        "done": True,
        "done_reason": "stop",
        "context": []
    }
    yield f"{json.dumps(final_chunk)}\n"

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    user_input = data.get('input')
    session_id = data.get('session_id', datetime.now().strftime("%Y%m%d%H%M%S"))
    conversation_started = data.get('conversation_started', True)
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    return Response(generate_stream(user_input, session_id, conversation_started), content_type='application/json')

@app.route('/api/tags', methods=['GET'])
def get_tags(): # placeholder to be compatible with ollama format
    tags = {"models": [{"name":"biomania",
                        "model":"biomania",
                        "modified_at":"2024-06-18T18:37:34.916232101-04:00",
                        "size":1,
                        "digest":"None",
                        "details":
                            {
                                "parent_model":"",
                                "format":"python-stream",
                                "family":"biomania",
                                "families": None,
                                "parameter_size":"None",
                                "quantization_level":"None"
                             }
                        }
                       ]
            }  # Replace with actual data fetching logic
    return jsonify(tags)

@app.route('/api/chat', methods=['POST'])
def chat():
    if request.is_json:
        data = request.json
    else:
        data = request.get_data(as_text=True)
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON"}), 400

    messages = data.get('messages')
    print(data)
    print(messages)
    if not messages or not isinstance(messages, list) or len(messages) == 0:
        return jsonify({"error": "No messages provided"}), 400
    
    user_input = messages[0].get('content')
    if not user_input:
        return jsonify({"error": "No content provided in the messages"}), 400
    
    session_id = data.get('session_id', datetime.now().strftime("%Y%m%d%H%M%S"))
    conversation_started = data.get('conversation_started', True)

    responses = process_input(model, user_input, library, session_id, conversation_started)
    output = []
    for response in responses:
        chunk = {
            "model": "biomania",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "message": {
                "role": "assistant",
                "content": response
            },
            "done": False
        }
        output.append(chunk)
    
    # Ensure the last response indicates completion
    final_chunk = {
        "model": "biomania",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "message": {
            "role": "assistant",
            "content": ""
        },
        "done": True,
        "done_reason": "stop",
        "context": []
    }
    output.append(final_chunk)
    return Response((f"{json.dumps(chunk)}\n" for chunk in output), content_type='application/json')

@app.route('/api/chat/biomania', methods=['POST'])
def chat_biomania():
    return chat()

if __name__ == '__main__':
    app.run(port=5000)

