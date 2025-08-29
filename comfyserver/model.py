import inspect
import random
from typing import Dict, List, Any, Callable, Tuple, Dict, Union
import torch
import gc
from .kutils import get_value_at_index
from nodes import NODE_CLASS_MAPPINGS

import numpy as np
from .api_mapper import InputOutputParser
from .api_mapper import convert_infer_input_to_comfyui

from kserve.protocol.infer_type import InferRequest, InferResponse,InferOutput,InferInput
from kserve import Model

from kserve.errors import InvalidInput

import asyncio
from threading import Thread


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
        NODE_CLASS_MAPPINGS = node_class_mappings
        self.visited = {}
        self.load_order = []
        self.is_special_function = False

    def determine_load_order(self) -> List[Tuple[str, Dict, bool]]:
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
            cls = NODE_CLASS_MAPPINGS[node_dict["class_type"]]()
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
            class_def = NODE_CLASS_MAPPINGS[self.data[key]["class_type"]]()
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


class ComfyModel(Model):
    """Executes a workflow directly instead of generating code."""

    def __init__(self, name: str, workflow: str):
        super().__init__(name)
        self.name = name
        self.workflow = workflow
        self.ready = False
        self._loading = False
        self._loader_outputs: dict[str, Any] = {}  # <- cached results

        load_order_determiner = LoadOrderDeterminer(workflow, NODE_CLASS_MAPPINGS)
        self.orders = load_order_determiner.determine_load_order()

        api_parser = InputOutputParser()
        self.infer_inputs, self.infer_outputs = api_parser.parse_api_data(self.workflow)
        self.input_names = [inp.name for inp in self.infer_inputs]
        self.output_names = [out.name for out in self.infer_outputs]

    async def get_input_types(self) -> List[Dict]:
        return [input_inf.to_dict() for input_inf in self.infer_inputs]

    async def get_output_types(self) -> List[Dict]:
        return [output_inf.to_dict() for output_inf in self.infer_outputs]

    # def load(self) -> bool:
    #     """Non-blocking load that starts background loading"""
    #     if not self._loading:
    #         self._loading = True
    #         # Start loading in background thread
    #         loading_thread = Thread(target=self._load_models, daemon=True)
    #         loading_thread.start()
    #     return True  # Return immediately
    
        
    def load(self) -> bool:
        "Execute all loader nodes exactly once and cache their results."
        # TODO Must do model seperation into different

        print(f"$$$$$$$$$ Loading models")
        initialized_objects = {}
        init_order, _, _ = self.orders

        cache_dict = {}
        with torch.no_grad():
            for idx, data, _ in init_order:
                cache_dict[idx] = {}
                
                cls_name = data["class_type"]
                cache_dict[idx]["class_type"] = cls_name

                # prepare inputs (nothing references loaders, so direct literals)
                inputs = self._process_inputs(data["inputs"], self._loader_outputs)
                cache_dict[idx]["inputs"] = inputs
                
                # instantiate (once per class)
                if cls_name not in initialized_objects:
                    initialized_objects[cls_name] = NODE_CLASS_MAPPINGS[cls_name]()
                
                obj = initialized_objects[cls_name]
                
                # TODO rewrite the caching logic to be more readable
                output = None
                for node_id, node_config in cache_dict.items():
                    if node_id == idx:
                        continue
                    if node_config["class_type"] == cls_name and node_config["inputs"] == inputs:
                        print(
                            f"[INIT] Reusing loader {node_id} ({cls_name}) for {idx} with same inputs {inputs}"
                        )
                        output = self._loader_outputs[node_id]
                        break
                    
                if not output:
                    print(
                    f"[INIT] Executed loader {idx} ({cls_name}) whith these inputs {inputs}"
                )
                    output = getattr(obj, obj.FUNCTION)(**inputs)
                    
                # run and cache
                self._loader_outputs[idx] = output
                # print(
                #     f"[INIT] Executed loader {idx} ({cls_name}) whith these inputs {inputs}"
                # )
        self.ready = True
        return self.ready

    async def preprocess(
        self, payload: InferRequest, headers: Dict[str, str] = None
    ) -> InferRequest:
        for infer_input in payload.inputs:
            if infer_input.name not in self.input_names:
                raise InvalidInput(
                    f"Input {infer_input.name} not found in model inputs."
                )

        if payload.request_outputs:
            for infer_output in payload.request_outputs:
                if infer_output.name not in self.output_names:
                    raise InvalidInput(
                        f"Output {infer_output.name} not found in model outputs."
                    )

        return payload

    async def postprocess(
        self,
        result: InferResponse,
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None,
    ) -> InferResponse:
        """The `postprocess` handler can be overridden for inference result or response transformation.
        The predictor sends back the inference result in `Dict` for v1 endpoints and `InferResponse` for v2 endpoints.

        Args:
            result: The inference result passed from `predict` handler or the HTTP response from predictor.
            headers: Request headers.

        Returns:
            A Dict or InferResponse after post-process to return back to the client.
        """
        return result

    async def predict(
        self, payload: InferRequest, headers: Dict[str, str] = None
    ) -> InferResponse:
        """Execute the workflow directly and return results."""

        _, runtime_order, _ = self.orders

        # Store initialized objects and executed variables
        initialized_objects = {}

        # Execute for the specified queue size
        print(f"Executing workflow")
        current_results = {}

        with torch.no_grad():
            # Process each node in the load order
            for idx, data, is_special_function in runtime_order:
                inputs, class_type = data["inputs"], data["class_type"]
                input_types = NODE_CLASS_MAPPINGS[class_type].INPUT_TYPES()

                # Check for missing required inputs
                missing_required_variable = False
                if "required" in input_types.keys():
                    for required in input_types["required"]:
                        if required not in inputs.keys():
                            missing_required_variable = True
                            break

                # prompt_input = prompt.get(idx,None)
                # prompt_input = prompt_input.get('inputs')

                if missing_required_variable:
                    print(
                        f"Skipping node {idx} ({class_type}) - missing required inputs"
                    )
                    continue

                # Skip preview image nodes
                if class_type == "PreviewImage":
                    print(f"Skipping PreviewImage node {idx}")
                    continue

                # Initialize the class if not already done
                if class_type not in initialized_objects:
                    initialized_objects[class_type] = NODE_CLASS_MAPPINGS[class_type]()
                    print(f"Initialized {class_type}")

                class_instance = initialized_objects[class_type]

                # Prepare inputs for execution
                processed_inputs = {}
                executed_variables = {**self._loader_outputs, **current_results}
                for key, value in inputs.items():
                    if isinstance(value, list) and len(value) >= 2:
                        # This is a reference to another node's output
                        node_id, output_index = value[0], value[1]
                        if node_id in executed_variables:
                            processed_inputs[key] = get_value_at_index(
                                executed_variables[node_id], output_index
                            )
                        else:
                            print(
                                f"Warning: Referenced node {node_id} not found in executed variables"
                            )
                            processed_inputs[key] = value
                    # TODO insert input replacement logic here
                    # TODO check for data type and convert if necessary
                    elif user_input:= payload.get_input_by_name(InputOutputParser.INFER_INPUT_NAME_TEMPLATE.format(
                        node_id=idx,
                        input_name=key
                    )):

                        processed_inputs[key] = convert_infer_input_to_comfyui(user_input) 
                        # processed_inputs[key] = user_input.data
                        
                    elif key in ["noise_seed", "seed"]:
                        # Generate random seed
                        processed_inputs[key] = random.randint(1, 2**64)
                    else:
                        processed_inputs[key] = value

                # processed_inputs = self._process_inputs(idx,inputs,{**self._loader_outputs, **current_results})

                # Add hidden variables if needed
                if (
                    "hidden" in input_types.keys()
                    and "unique_id" in input_types["hidden"].keys()
                ):
                    processed_inputs["unique_id"] = random.randint(1, 2**64)
                elif hasattr(class_instance, class_instance.FUNCTION):
                    func_params = self._get_function_parameters(
                        getattr(class_instance, class_instance.FUNCTION)
                    )
                    if func_params and "unique_id" in func_params:
                        processed_inputs["unique_id"] = random.randint(1, 2**64)

                # Filter inputs to only include valid parameters
                if hasattr(class_instance, class_instance.FUNCTION):
                    func_params = self._get_function_parameters(
                        getattr(class_instance, class_instance.FUNCTION)
                    )
                    if func_params is not None:
                        processed_inputs = {
                            key: value
                            for key, value in processed_inputs.items()
                            if key in func_params
                        }

                try:
                    # Execute the function
                    result = getattr(class_instance, class_instance.FUNCTION)(
                        **processed_inputs
                    )
                    current_results[idx] = result
                    print(f"Executed node {idx} ({class_type}) successfully")

                except Exception as e:
                    print(f"Error executing node {idx} ({class_type}): {str(e)}")
                    current_results[idx] = None

            # Create response outputs with actual data
        response_outputs = self.create_infer_outputs_for_response(current_results, payload)
        

    # TODO Cleen this part
        gc.collect()

        if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
            
        
        return InferResponse(
            response_id=payload.id,
            model_name=self.name,
            infer_outputs=response_outputs,
            parameters={"execution_successful": True},
        )

    def _process_inputs(self,inputs: Dict, executed_variables: Dict) -> Dict:
        """Process inputs, replacing references with actual values."""
        # TODO add a part which replaces the inputs form incumming request mabe in pre-predict
        processed_inputs = {}

        for key, value in inputs.items():
            if isinstance(value, list) and len(value) >= 2:
                # This is a reference to another node's output
                node_id, output_index = value[0], value[1]
                if node_id in executed_variables:
                    processed_inputs[key] = get_value_at_index(
                        executed_variables[node_id], output_index
                    )
                else:
                    print(
                        f"Warning: Referenced node {node_id} not found in executed variables"
                    )
                    processed_inputs[key] = value
            # TODO insert input replacement logic here
            elif key in ["noise_seed", "seed"]:
                # Generate random seed
                processed_inputs[key] = random.randint(1, 2**64)
            else:
                processed_inputs[key] = value

        return processed_inputs

    def _get_function_parameters(self, func: Callable) -> List:
        """Get the names of a function's parameters."""
        try:
            signature = inspect.signature(func)
            parameters = {
                name: param.default if param.default != param.empty else None
                for name, param in signature.parameters.items()
            }

            catch_all = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )

            return list(parameters.keys()) if not catch_all else None
        except:
            return None
    
    def create_infer_outputs_for_response(self, current_results: Dict, payload: InferRequest) -> List[InferOutput]:
        """
        Create InferOutput objects with actual data for the prediction response.
        Uses the shape and type specifications from self.infer_outputs but populates with actual results.
        
        Args:
            current_results: Dictionary containing the execution results from workflow nodes
            payload: The original request payload to check which outputs are requested
            
        Returns:
            List of InferOutput objects with actual data matching the expected specifications
        """
        response_outputs = []
        
        # Get requested outputs from payload, or use all if none specified
        requested_output_names = []
        if payload.request_outputs:
            requested_output_names = [req_out.name for req_out in payload.request_outputs]
        else:
            return response_outputs
            requested_output_names = [out.name for out in self.infer_outputs]
        
        for output_spec in self.infer_outputs:
            # Skip if this output wasn't requested
            if output_spec.name not in requested_output_names:
                continue

            node_id, output_index, output_type = output_spec.name.split("_", 2)
            # TODO add two options for specifying output nme 

            # node_id = output_spec.parameters.get("node_id")
            # output_index = output_spec.parameters.get("output_index")
            # output_type = output_spec.parameters.get("output_type")
            
            if node_id is None or output_index is None or output_type is None:
                continue
                

            # Get the actual result data from workflow execution
            if node_id in current_results and current_results[node_id] is not None:
                result_data = get_value_at_index(current_results[node_id], int(output_index))
                
                # Convert the result data to the appropriate format for KServe
                converted_data = self._convert_comfyui_output_to_kserve(result_data, output_type, output_spec.shape)
                
                # Create new InferOutput with actual data
                response_output = InferOutput(
                    name=output_spec.name,
                    shape=list(converted_data.shape) if hasattr(converted_data, 'shape') else output_spec.shape,
                    datatype=output_spec.datatype,
                    parameters=output_spec.parameters
                )
                
                # Set the actual data
                # response_output.data = converted_data.tolist() if hasattr(converted_data, 'tolist') else [converted_data]
                response_output.data = converted_data
                response_outputs.append(response_output)
        
        return response_outputs

    # def _convert_comfyui_output_to_kserve(self, data: Any, output_type: str, expected_shape: List[int]) -> Any:
    #     """
    #     Convert ComfyUI output data to KServe-compatible format.
        
    #     Args:
    #         data: Raw output data from ComfyUI node execution
    #         output_type: The expected output type
    #         expected_shape: The expected shape for the output
            
    #     Returns:
    #         Converted data in appropriate format
    #     """
    #     # Handle different output types
    #     if output_type in ["IMAGE", "MASK", "LATENT", "VIDEO", "NOISE"]:
    #         # Convert tensor-like data to numpy arrays
    #         if hasattr(data, 'cpu'):  # PyTorch tensor
    #             return data.cpu().numpy()
    #         elif hasattr(data, 'numpy'):  # Other tensor types
    #             return data.numpy()
    #         elif isinstance(data, np.ndarray):
    #             return data
    #         else:
    #             return np.array(data)
                
    #     elif output_type in ["STRING"]:
    #         # Convert to bytes for BYTES datatype
    #         if isinstance(data, str):
    #             return data.encode('utf-8')
    #         return str(data).encode('utf-8')
            
    #     elif output_type in ["INT", "FLOAT", "NUMBER", "BOOLEAN"]:
    #         # Scalar values
    #         return np.array([data])
            
    #     elif output_type in ["POINT", "BBOX"]:
    #         # Convert to numpy array
    #         return np.array(data)
            
    #     else:
    #         # Default: try to convert to numpy array or return as-is
    #         try:
    #             return np.array(data)
    #         except:
    #             return data
    def _convert_comfyui_output_to_kserve(self, data: Any, output_type: str, expected_shape: List[int]) -> Any:
        # TODO compare with other implemetation
        
        """
        Convert ComfyUI output data to KServe-compatible format with CPU tensors.
        
        Args:
            data: Raw output data from ComfyUI node execution
            output_type: The expected output type
            expected_shape: The expected shape for the output
            
        Returns:
            Converted data in appropriate format (all tensors moved to CPU)
        """
        
        # Helper function to recursively move tensors to CPU and detach from computation graph
        def to_cpu(obj):
            if hasattr(obj, 'is_cuda') and obj.is_cuda:  # PyTorch tensor on GPU
                return obj.detach().to("cpu", non_blocking=False).contiguous()
            elif hasattr(obj, 'cpu'):  # Any tensor with .cpu() method
                return obj.cpu().detach() if hasattr(obj, 'detach') else obj.cpu()
            elif isinstance(obj, (list, tuple)):
                return type(obj)(to_cpu(x) for x in obj)
            elif isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            return obj
        
        # Convert input data to CPU first - this is the key addition
        data = to_cpu(data)
        
        # Handle different output types (same logic as before, but now working with CPU data)
        if output_type in ["IMAGE", "MASK", "LATENT", "VIDEO", "NOISE"]:
            # Convert tensor-like data to numpy arrays
            if hasattr(data, 'cpu'):  # PyTorch tensor (should now be on CPU)
                return data.cpu().numpy()
            elif hasattr(data, 'numpy'):  # Other tensor types
                return data.numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                return np.array(data)
                
        elif output_type in ["STRING"]:
            # Convert to bytes for BYTES datatype
            if isinstance(data, str):
                return data.encode('utf-8')
            return str(data).encode('utf-8')
            
        elif output_type in ["INT", "FLOAT", "NUMBER", "BOOLEAN"]:
            # Scalar values
            return np.array([data])
            
        elif output_type in ["POINT", "BBOX"]:
            # Convert to numpy array
            return np.array(data)
            
        else:
            # Default: try to convert to numpy array or return as-is
            try:
                return np.array(data)
            except:
                return data


