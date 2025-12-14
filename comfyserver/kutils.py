import json
import logging
import os
import sys
from typing import Any, Dict, List, Mapping, Sequence, TextIO, Tuple, Union

import folder_paths
import yaml

# TODO loggers nodes
# TODO Write input and output node path 
def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        # custom_nodes_dependency
        if "base_path" in conf:
            base_path = conf.pop("base_path")
            base_path = os.path.expandvars(os.path.expanduser(base_path))
            if not os.path.isabs(base_path):
                base_path = os.path.abspath(os.path.join(yaml_dir, base_path))
        # if "custom_nodes_dependency" in conf:
        #     custom_nodes_dependency = conf.pop("custom_nodes_dependency")
        #     custom_nodes_dependency = os.path.expandvars(os.path.expanduser(custom_nodes_dependency))
        #     if not os.path.isabs(custom_nodes_dependency):
        #         base_path = os.path.abspath(os.path.join(yaml_dir, custom_nodes_dependency))
        #     sys.path.insert(-1, custom_nodes_dependency)
        is_default = False
        if "is_default" in conf:
            is_default = conf.pop("is_default")
        
        if is_default:
            # Set default input and output directories if the falg is set
            folder_paths.models_dir = os.path.join(str(base_path), "models")
            folder_paths.output_directory = os.path.join(str(base_path), "output")
            folder_paths.temp_directory = os.path.join(str(base_path), "temp")
            folder_paths.input_directory = os.path.join(str(base_path), "input")
            folder_paths.user_directory = os.path.join(str(base_path), "user")
        
        for dir in ["models","output","temp","input","user"]:
            dir_name = f"default_{dir}_dir"
            if dir_name in conf:
                dir_path = conf.pop(dir_name)
                if dir == "models":
                    folder_paths.models_dir = os.path.join(str(base_path), str(dir))
                else:
                    setattr(folder_paths,f"{dir}_directory",os.path.join(str(dir_path), ""))            
        

            
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
    
class LoadOrderDeterminer:
    """Determine the load order of each key in the provided dictionary.

    This class places the nodes without node dependencies first, then ensures that any node whose
    result is used in another node will be added to the list in the order it should be executed.

    Attributes:
        data (Dict): The dictionary for which to determine the load order.
        node_class_mappings (Dict): Mappings of node classes.
    """
    _LOADER_CATEGORIES = ["loaders", "advanced/loaders"]

    def __init__(self, data: Dict, node_class_mappings: Dict):
        """Initialize the LoadOrderDeterminer with the given data and node class mappings.

        Args:
            data (Dict): The dictionary for which to determine the load order.
            node_class_mappings (Dict): Mappings of node classes.
        """
        self.data = data
        self.node_class_mappings = node_class_mappings
        self.visited = {}
        self.load_order = []
        self.is_special_function = False

    def determine_load_order(self) -> Tuple[List,List,List]:
        """Determine the load order for the given data.

        Returns:
            List[Tuple[str, Dict, bool]]: A list of tuples representing the load order.
        """
        init_order, runtime_order = [], []

        # visit in topological order (same DFS you already have)
        self._load_special_functions_first()  # ensures loader nodes visited first
        self.is_special_function = False

        for key in self.data:
            if key not in self.visited:
                self._dfs(key)

        # ➊  split while preserving relative order
        for entry in self.load_order:
            _key, node_dict, _ = entry
            cls = self.node_class_mappings[node_dict["class_type"]]()
            (init_order if cls.CATEGORY in self._LOADER_CATEGORIES else runtime_order).append(entry)

        return init_order, runtime_order, self.load_order

    def _dfs(self, key: str) -> None:
        """Depth-First Search function to determine the load order.

        Args:
            key (str): The key from which to start the DFS.

        Returns:
            None
        """
        # Mark the node as visited.
        self.visited[key] = True
        inputs = self.data[key]["inputs"]
        # Loop over each input key.
        for input_key, val in inputs.items():
            # If the value is a list and the first item in the list has not been visited yet,
            # then recursively apply DFS on the dependency.
            if isinstance(val, list) and val[0] not in self.visited:
                self._dfs(val[0])
        # Add the key and its corresponding data to the load order list.
        self.load_order.append((key, self.data[key], self.is_special_function))

    def _load_special_functions_first(self) -> None:
        """Load functions without dependencies, loaderes, and encoders first.

        Returns:
            None
        """
        # Iterate over each key in the data to check for loader keys.
        for key in self.data:
            class_def = self.node_class_mappings[self.data[key]["class_type"]]()
            # Check if the class is a loader class or meets specific conditions.
            if (
                class_def.CATEGORY == "loaders"
                or class_def.FUNCTION in ["encode"]
                or not any(
                    isinstance(val, list) for val in self.data[key]["inputs"].values()
                )
            ):
                self.is_special_function = True
                # If the key has not been visited, perform a DFS from that key.
                if key not in self.visited:
                    self._dfs(key)