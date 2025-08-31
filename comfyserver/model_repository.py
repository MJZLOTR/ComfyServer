import asyncio
import os
import threading
from typing import Dict, Optional

from kserve.model_repository import ModelRepository

from .kutils import FileHandler
from .model import ComfyModel


class ConfyModelRepository(ModelRepository):

    def __init__(self, workflow_dir: str):
        super().__init__(models_dir=workflow_dir)
        self.models_load_corutines = {} 
        self.load_models()

    async def load(self, name: str):
        if name not in self.models_load_corutines:
            self.models_load_corutines[name] = asyncio.create_task(
                asyncio.to_thread(self.load_model, name)
            )
        
        return self.models_load_corutines[name]
        # return self.load_model(name)

    def load_models(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # get all json files in the models_dir
        for file in os.listdir(self.models_dir):
            if file.endswith(".json"):
                model_name = os.path.splitext(file)[0]
                self.models_load_corutines[model_name] = loop.create_task(
                asyncio.to_thread(self.load_model, model_name)
            )
        # TODO: is written messyly, refactor
        if self.models_load_corutines:
            def run_coroutines_in_thread():
                try:
                    # Run all coroutines concurrently
                    coroutines = list(self.models_load_corutines.values())
                    results = loop.run_until_complete(asyncio.gather(*coroutines))
                    return results
                finally:
                    loop.close()
            
            # Start the thread
            thread = threading.Thread(target=run_coroutines_in_thread)
            thread.daemon = True  # Optional: makes thread die when main program exits
            thread.start()
    

    def load_model(self, workflow_name) -> bool:
        workflow_addr = os.path.join(self.models_dir, f"{workflow_name}.json")
        workflow = FileHandler.read_json_file(workflow_addr)
        model = ComfyModel(workflow_name, workflow)

        self.update(model)
        if not model.load():
            self.unload(workflow_name)
        return model.ready

    def unload(self, name: str):
        super().unload(name)
        self.models_load_corutines.pop(name, None)