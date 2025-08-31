import json
import os
from platform import node
from typing import List, Dict, Any, Tuple, Optional, Union
from kserve import InferInput, InferOutput
import numpy as np

from nodes import NODE_CLASS_MAPPINGS
from comfy.comfy_types import IO


def convert_infer_input_to_comfyui(infer_input: InferInput) -> Any:
    """
    Convert incoming InferInput data to a type acceptable for ComfyUI node input.
    
    Args:
        infer_input: An object representing InferInput that includes:
                     - data: the incoming data (list, bytes, numpy array)
                     - shape: shape of the incoming data
                     
    Returns:
        Converted data for ComfyUI input
        
    Examples:
        data: [42], shape: [1] => 42
        data: [b'test'], shape: [1] => 'test'  
        data: numpy_array, shape: [H,W,C] => numpy_array
    """
    data = infer_input.as_numpy()
    shape = infer_input.shape
    # data = getattr(infer_input, 'data', None)
    # shape = getattr(infer_input, 'shape', None)
    
    
    # if not data:
    #     return None
    
    # Handle scalar values (including bytes that need decoding)
    if shape == [1]:
        val = data[0]
        # If bytes, decode to string
        if isinstance(val, (bytes, bytearray)):
            return val.decode('utf-8')
        return val
    
    # Handle numpy arrays - pass through as-is
    if isinstance(data, np.ndarray):
        return data
    
    # Handle list to numpy array conversion for multi-dimensional data
    if isinstance(data, list) and shape and len(shape) > 1:
        np_array = np.array(data)
        # If shape has -1 (flexible dimension), skip strict reshape
        if -1 not in shape and list(np_array.shape) != shape:
            try:
                np_array = np_array.reshape(shape)
            except Exception as e:
                raise ValueError(f"Cannot reshape data from shape {np_array.shape} to expected shape {shape}: {e}")
        return np_array
    
    # Fallback: return raw data
    return data


class InputOutputParser:
    """
    Parser for ComfyUI API JSON files using actual ComfyUI nodes and NODE_CLASS_MAPPINGS
    """
    INFER_INPUT_NAME_TEMPLATE = "{node_id}_{input_name}"
    INFER_OUTPUT_NAME_TEMPLATE = "{node_id}_{output_index}_{output_type}"

    _TYPE_MAPPING = {
        IO.INT: "INT32",
        IO.FLOAT: "FP32",
        IO.NUMBER: "FP32",
        IO.STRING: "BYTES",
        IO.BOOLEAN: "BOOL",
        IO.IMAGE: "FP32",
        IO.MASK: "FP32",
        IO.LATENT: "FP32",

        IO.CONDITIONING: "FP32",
        IO.CLIP: "FP32",
        IO.CLIP_VISION: "FP32",
        IO.CLIP_VISION_OUTPUT: "FP32",
        IO.NOISE: "FP32",
        IO.SIGMAS: "FP32",
        IO.AUDIO: "FP32",
        IO.VIDEO: "FP32",
        IO.POINT: "FP32",
        IO.FACE_ANALYSIS: "FP32",
        IO.BBOX: "FP32",
        IO.SEGS: "FP32",
        IO.WEBCAM: "FP32",
   }
    _SAPE_MAPPING = {
    IO.INT: [1],
    IO.FLOAT: [1], 
    IO.NUMBER: [1], 
    IO.STRING: [1],
    IO.BOOLEAN: [1],
    IO.IMAGE: [-1, -1, -1, 3],
    IO.MASK: [-1, -1, -1],
    IO.LATENT: [-1, 4, -1, -1],
    IO.VIDEO: [-1, -1, -1, -1, 3],  # More flexible time dimension
    IO.POINT: [1, 2],
    IO.BBOX: [1, 4],
    
    # IO.CONDITIONING: [-1, -1, -1],     # Flexible seq_len and hidden_dim
    IO.NOISE: [-1, -1, -1, 3],        # Match target dimensions
    IO.SIGMAS: [-1, -1, -1],          # Flexible dimensions
    IO.CLIP: [1, -1],                 # Flexible embedding size
    IO.CLIP_VISION: [1, -1],          
    IO.CLIP_VISION_OUTPUT: [1, -1],   
    IO.AUDIO: [1, -1],                # Flexible audio length
    IO.FACE_ANALYSIS: [1, -1],        # Flexible feature size
    IO.SEGS: [-1, -1, -1],            # Flexible resolution
    IO.WEBCAM: [-1, -1, -1, 3],       # Flexible resolution
    }
    _IGNORE_TYPES = {
        "IGNORE",IO.ANY, IO.COMBO, IO.SAMPLER, IO.GUIDER, IO.CONTROL_NET,
        IO.VAE, IO.MODEL, IO.LORA_MODEL, IO.LOSS_MAP, IO.STYLE_MODEL,
        IO.GLIGEN, IO.UPSCALE_MODEL, IO.PRIMITIVE, IO.CONDITIONING,
    }

    def get_desired_infer_output(self, node_id: str, output_index: int, output_type: str) -> InferOutput:
        """
        Generate an InferOutput object for the given node output, using the
        shape and datatype mappings already defined in the class.

        Args:
            node_id: Identifier of the node
            output_index: Index of the output in the node's output list
            output_type: Output type string (e.g., from ComfyUI IO types)

        Returns:
            An InferOutput object with appropriate name, shape, datatype, and parameters
        """
        # Determine KServe datatype using existing mapping
        kserve_dtype = self._TYPE_MAPPING.get(output_type, "BYTES")
        
        # Determine shape using existing mapping  
        shape = self._SAPE_MAPPING.get(output_type, [-1])
        
        # Create metadata parameters
        parameters = {
            "output_index": output_index,
            "node_id": node_id,
            "output_type": output_type
        }
        
        # Generate output name using existing template
        output_name = self.INFER_OUTPUT_NAME_TEMPLATE.format(
            node_id=node_id,
            output_index=output_index,
            output_type=output_type
        )
        
        # Create and return InferOutput object
        infer_output = InferOutput(
            name=output_name,
            shape=shape,
            datatype=kserve_dtype,
            parameters=parameters
        )
        
        return infer_output


    
    def _extract_type_from_input_spec(self, input_spec: Union[Tuple, List, str]) -> str:
        """
        Extract type string from ComfyUI input specification
        
        Args:
            input_spec: ComfyUI input spec like (IO.STRING, {...}) or ("COMBO", [...])
        
        Returns:
            Type string for KServe mapping
        """
        if isinstance(input_spec, (tuple, list)) and input_spec:
            return str(input_spec[0])
        elif isinstance(input_spec, str):
            return input_spec
        else:
            return IO.ANY  # Default fallback
    
    def _get_node_io_info(self, class_type: str) -> Tuple[Optional[type], Dict[str, str], Tuple[str, ...]]:
        """
        Get input and output type information from ComfyUI node class
        
        Args:
            class_type: ComfyUI node class name
            
        Returns:
            Tuple of (input_types_dict, output_types_tuple)
        """
        node_class = NODE_CLASS_MAPPINGS.get(class_type)
        if node_class is None:
            raise KeyError(f"Unknown node class type: {class_type}")
        
        # Get input types
        input_types_func = getattr(node_class, "INPUT_TYPES", None)
        input_required_types = {}
        if callable(input_types_func):
            try:
                input_types_dict = input_types_func()
                required_inputs = input_types_dict.get("required", {})
                
                # Extract type information from each input specification
                for input_name, input_spec in required_inputs.items():
                    input_required_types[input_name] = self._extract_type_from_input_spec(input_spec)
            except Exception as e:
                print(f"Error extracting input types for {class_type}: {e}")
        
        # Get output types
        output_types = getattr(node_class, "RETURN_TYPES", ())
        
        return node_class,input_required_types, output_types
    
    def parse_api_data(self, api_data: Dict[str, Any]) -> Tuple[List[InferInput], List[InferOutput]]:
        """
        Parse ComfyUI API data and generate InferInput/InferOutput lists
        
        Args:
            api_data: Parsed ComfyUI API JSON data
            
        Returns:
            Tuple of (infer_inputs, infer_outputs)
        """
        infer_inputs = []
        infer_outputs = []
        
        for node_id, node_data in api_data.items():
            class_type = node_data.get("class_type")
            inputs = node_data.get("inputs", {})
            
                        
            try:
                # Get node type information from ComfyUI
                node_class, input_types, output_types = self._get_node_io_info(class_type)
            except KeyError as e:
                print(f"Warning: {e}, skipping node {node_id}")
                continue
            except Exception as e:
                print(f"Error processing node {class_type} (ID: {node_id}): {e}")
                continue
            
            if node_class.CATEGORY == "loaders":
                continue

            node_metadata = {
                "node_id": node_id,
                "node_title": node_data.get("_meta", {}).get("title", ""),
                "class_type": class_type,
                "class_name": node_class.__name__,
            }
            # Process inputs
            for input_name, input_value in inputs.items():
                # Skip inputs that are outputs from other nodes
                if  isinstance(input_value, list) and len(input_value) >= 2:
                    continue
                
                # Get input type from node definition
                input_type = input_types.get(input_name, IO.ANY)

                if input_type in self._IGNORE_TYPES:
                    print(f"Skipping ignored input type: {input_type} for input {input_name} in node {node_id}")
                    continue
                
                # Convert to KServe data type
                kserve_dtype = self._TYPE_MAPPING.get(input_type, "BYTES")
                
                # Determine shape
                shape = self._SAPE_MAPPING.get(input_type, [-1])
                
                # Create InferInput
                infer_input = InferInput(
                    name= self.INFER_INPUT_NAME_TEMPLATE.format(
                        node_id=node_id,
                        input_name=input_name
                    ),
                    shape=shape,
                    datatype=kserve_dtype,
                    parameters=node_metadata
                )
                
                
                infer_inputs.append(infer_input)
            
            # Process outputs
            for idx, output_type in enumerate(output_types):
                if output_type in self._IGNORE_TYPES:
                    print(f"Skipping ignored output type: {output_type} for node {node_id}")
                    continue
                
                # Convert to KServe data type
                kserve_dtype = self._TYPE_MAPPING.get(output_type, "BYTES")
                shape = self._SAPE_MAPPING.get(output_type, [-1])
                
                infer_output = InferOutput(
                    name=self.INFER_OUTPUT_NAME_TEMPLATE.format(
                        node_id=node_id,
                        output_index=idx,
                        output_type=output_type
                    ),
                    shape=shape,
                    datatype=kserve_dtype,
                    parameters={
                        "output_index": idx,
                        **node_metadata
                    }
                )
                
                infer_outputs.append(infer_output)
        
        return infer_inputs, infer_outputs


# Example usage
if __name__ == "__main__":
    parser = InputOutputParser()
    wf = json.load(open("/home/ubuntu/workspace/test_workflows/simple_wf.json", "r"))
    i,o =parser.parse_api_data(wf)
    print(i,o)

