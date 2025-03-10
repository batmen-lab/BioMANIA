# [BIOAGENT]
from .model import Model
import logging

if __name__ == "__main__":
    conversation_started = True
    logger = logging.getLogger(__name__)
    model = Model(logger=logger, device='cpu')
    user_input = "Could you load the built in dataset?"
    library = "scanpy"
    model.run_pipeline(user_input, library, top_k=1, files=[], conversation_started=conversation_started, session_id="")