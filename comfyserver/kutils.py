import os
from typing import Sequence, Mapping, Any, Union,TextIO
import sys
import json

import yaml
import folder_paths
import logging

def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
            base_path = os.path.expandvars(os.path.expanduser(base_path))
            if not os.path.isabs(base_path):
                base_path = os.path.abspath(os.path.join(yaml_dir, base_path))
        is_default = False
        if "is_default" in conf:
            is_default = conf.pop("is_default")
        
        if is_default:
            # Set default input and output directories if the falg is set
            folder_paths.output_directory = os.path.join(base_path, "output")
            folder_paths.temp_directory = os.path.join(base_path, "temp")
            folder_paths.input_directory = os.path.join(base_path, "input")
            folder_paths.user_directory = os.path.join(base_path, "user")
            
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path:
                    full_path = os.path.join(base_path, full_path)
                elif not os.path.isabs(full_path):
                    full_path = os.path.abspath(os.path.join(yaml_dir, y))
                normalized_path = os.path.normpath(full_path)
                logging.info("Adding extra search path {} {}".format(x, normalized_path))
                folder_paths.add_model_folder_path(x, normalized_path, is_default)
                
        


class FileHandler:
    """Handles reading and writing files.

    This class provides methods to read JSON data from an input file and write code to an output file.
    """

    @staticmethod
    def read_json_file(file_path: str | TextIO, encoding: str = "utf-8") -> dict:
        """
        Reads a JSON file and returns its contents as a dictionary.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            dict: The contents of the JSON file as a dictionary.

        Raises:
            FileNotFoundError: If the file is not found, it lists all JSON files in the directory of the file path.
            ValueError: If the file is not a valid JSON.
        """

        if hasattr(file_path, "read"):
            return json.load(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


class DummyMeta(type):
    def __getattr__(cls, name):
        # Handle class-level attribute access (e.g., DummyObject.instance)
        return DummyObject()

class DummyObject(metaclass=DummyMeta):
    def __init__(self, *args, **kwargs):
        # Store any arguments as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __getattr__(self, name):
        # Return a new DummyObject for any instance attribute access
        return DummyObject()
    
    def __call__(self, *args, **kwargs):
        # Return a new DummyObject for any method calls
        return DummyObject()
    
    def __getitem__(self, key):
        # Handle indexing operations
        return DummyObject()
    
    def __setitem__(self, key, value):
        # Handle setting items
        pass
    
    def __iter__(self):
        # Handle iteration
        return iter([])
