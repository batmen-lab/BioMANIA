from typing import Any, Dict, List, Union
import queue
class ServerEventCallback():
    """Base callback handler"""

    def __init__(self, queue: queue.Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue
        self.llm_block_id = 0
        self.tool_block_id = 0

    def add_to_queue(self, method_name: str, block_id, **kwargs: Any):
        data = {
            "method_name": method_name,
            "block_id": block_id,
        }
        data.update(kwargs)
        self.queue.put(data)

    def on_tool_retrieval_start(self):
        # e.g. [callback.on_tool_retrieval_start() for callback in self.callbacks]
        self.add_to_queue(
            "on_tool_retrieval_start",
            "recommendation-1",
        )
        print("on_tool_retrieval_start method called")

    def on_tool_retrieval_end(self, tools):
        # [callback.on_tool_retrieval_end(tools=[{'name':self.predicted_api_name, 'parameters':self.API_composite[self.predicted_api_name]['Parameters'],'description':json.dumps(self.API_composite[self.predicted_api_name],indent=4)}]) for callback in self.callbacks]
        self.add_to_queue(
            "on_tool_retrieval_end",
            "recommendation-1",
            recommendations=tools
        )
        print("on_tool_retrieval_end method called")
    
    def on_agent_action(self, block_id, task, task_title="",composite_code="",depth=0,color="black",imageData="",tableData="") -> str:
        # e.g. [callback.on_agent_action(block_id="textcollapse-" + str(self.indexxxx),task=response,) for callback in self.callbacks]
        self.tool_block_id += 1
        self.add_to_queue(
            "on_agent_action",
            block_id=block_id,
            task = task,
            task_title=task_title,
            composite_code=composite_code,
            imageData=imageData,
            tableData=tableData,
            depth=depth,
            color=color,
        )
        print("on_agent_action method called")
        return block_id

    def on_tool_start(self, api_name: str = "", api_calling: str = "", api_description: str = "", depth: int = 0) -> Any:
        method_name = "on_tool_start"
        self.add_to_queue(
            method_name,
            block_id="tool-" + str(self.tool_block_id),
            api_name=api_name,
            api_description=api_description,
            api_calling=api_calling,
            depth=depth
        )
        print("on_tool_start method called")

    def on_tool_end(self, task:str="", status:str="0", depth: int=0) -> Any:
        method_name = "on_tool_end"
        self.add_to_queue(
            method_name,
            block_id="tool-" + str(self.tool_block_id),
            task=task,
            status=status,
            depth=depth
        )
        print("on_tool_end method called")

    def on_agent_end(self, block_id:str, depth: int):
        self.add_to_queue(
            "on_agent_end",
            block_id=block_id,
            depth=depth
        )
        print("on_agent_end method called")