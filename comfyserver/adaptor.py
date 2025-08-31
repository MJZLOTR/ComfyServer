from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from comfy.comfy_types import IO
from kserve import InferInput, InferOutput, InferRequest, InferResponse
from nodes import NODE_CLASS_MAPPINGS

from .kutils import get_value_at_index


class ComfyKserveMapper:
    """
    Parser for ComfyUI API JSON files using actual ComfyUI nodes and NODE_CLASS_MAPPINGS
    """

    # Naming templates for inputs infer object based on node ID and input name
    INFER_INPUT_NAME_TEMPLATE = "{node_id}_{input_name}"

    # Naming templates for outputs infer object based on node ID, output index, and output type
    INFER_OUTPUT_NAME_TEMPLATE = "{node_id}_{output_index}_{output_type}"

    # Mapping from ComfyUI IO types to datatypes accepted by KServe
    _TYPE_MAPPING = {
        IO.INT: "INT32",
        IO.FLOAT: "FP32",
        IO.NUMBER: "FP32",
        IO.STRING: "BYTES",
        IO.BOOLEAN: "BOOL",
        IO.IMAGE: "FP32",
        IO.MASK: "FP32",
        IO.LATENT: "FP32",
        # TODO remove not supported types and add them to ignore list
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

    # Default shape mappings for ComfyUI IO types to KServe shapes
    _SHAPE_MAPPING = {
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
        IO.NOISE: [-1, -1, -1, 3],  # Match target dimensions
        IO.SIGMAS: [-1, -1, -1],  # Flexible dimensions
        IO.CLIP: [1, -1],  # Flexible embedding size
        IO.CLIP_VISION: [1, -1],
        IO.CLIP_VISION_OUTPUT: [1, -1],
        IO.AUDIO: [1, -1],  # Flexible audio length
        IO.FACE_ANALYSIS: [1, -1],  # Flexible feature size
        IO.SEGS: [-1, -1, -1],  # Flexible resolution
        IO.WEBCAM: [-1, -1, -1, 3],  # Flexible resolution
    }

    # Types to ignore during input/output processing
    _IGNORE_TYPES = {
        "IGNORE",
        IO.ANY,
        IO.COMBO,
        IO.SAMPLER,
        IO.GUIDER,
        IO.CONTROL_NET,
        IO.VAE,
        IO.MODEL,
        IO.LORA_MODEL,
        IO.LOSS_MAP,
        IO.STYLE_MODEL,
        IO.GLIGEN,
        IO.UPSCALE_MODEL,
        IO.PRIMITIVE,
        IO.CONDITIONING,
    }

    @staticmethod
    def _extract_type_from_input_spec(input_spec: Union[Tuple, List, str]) -> str:
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

    @classmethod
    def _get_node_io_info(
        cls, class_type: str
    ) -> Tuple[Optional[type], Dict[str, str], Tuple[str, ...]]:
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
                    input_required_types[input_name] = (
                        cls._extract_type_from_input_spec(input_spec)
                    )
            except Exception as e:
                print(f"Error extracting input types for {class_type}: {e}")

        # Get output types
        output_types = getattr(node_class, "RETURN_TYPES", ())

        return node_class, input_required_types, output_types

    @classmethod
    def convert_workflow_to_inference_objects(
        cls, workflow: Dict[str, Any]
    ) -> Tuple[List[InferInput], List[InferOutput]]:
        """
        Parse ComfyUI API data and generate InferInput/InferOutput lists

        Args:
            workflow: Parsed ComfyUI API JSON data

        Returns:
            Tuple of (infer_inputs, infer_outputs)
        """
        infer_inputs = []
        infer_outputs = []

        for node_id, node_data in workflow.items():
            class_type = node_data.get("class_type")
            inputs = node_data.get("inputs", {})

            try:
                # Get node type information from ComfyUI
                node_class, input_types, output_types = cls._get_node_io_info(
                    class_type
                )
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
                if isinstance(input_value, list) and len(input_value) >= 2:
                    continue

                # Get input type from node definition
                input_type = input_types.get(input_name, IO.ANY)

                if input_type in cls._IGNORE_TYPES:
                    print(
                        f"Skipping ignored input type: {input_type} for input {input_name} in node {node_id}"
                    )
                    continue

                # Convert to KServe data type
                kserve_dtype = cls._TYPE_MAPPING.get(input_type, "BYTES")

                # Determine shape
                shape = cls._SHAPE_MAPPING.get(input_type, [-1])

                # Create InferInput
                infer_input = InferInput(
                    name=cls.INFER_INPUT_NAME_TEMPLATE.format(
                        node_id=node_id, input_name=input_name
                    ),
                    shape=shape,
                    datatype=kserve_dtype,
                    parameters=node_metadata,
                )

                infer_inputs.append(infer_input)

            # Process outputs
            for idx, output_type in enumerate(output_types):
                if output_type in cls._IGNORE_TYPES:
                    print(
                        f"Skipping ignored output type: {output_type} for node {node_id}"
                    )
                    continue

                # Convert to KServe data type
                kserve_dtype = cls._TYPE_MAPPING.get(output_type, "BYTES")
                shape = cls._SHAPE_MAPPING.get(output_type, [-1])

                infer_output = InferOutput(
                    name=cls.INFER_OUTPUT_NAME_TEMPLATE.format(
                        node_id=node_id, output_index=idx, output_type=output_type
                    ),
                    shape=shape,
                    datatype=kserve_dtype,
                    parameters={"output_index": idx, **node_metadata},
                )

                infer_outputs.append(infer_output)

        return infer_inputs, infer_outputs

    @classmethod
    def convert_inferInput_to_comfy(cls,infer_input: InferInput) -> Any:
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

        # Handle scalar values (including bytes that need decoding)
        if shape == [1]:
            val = data[0]
            # If bytes, decode to string
            if isinstance(val, (bytes, bytearray)):
                return val.decode("utf-8")
            else:
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
                    raise ValueError(
                        f"Cannot reshape data from shape {np_array.shape} to expected shape {shape}: {e}"
                    )
            return np_array

        # Fallback: return raw data
        return data

    @classmethod
    def generate_inference_response_outputs(
        cls, node_results: Dict, infer_request: InferRequest,infer_outputs: List[InferOutput]
    ) -> List[InferOutput]:
        """
        Create InferOutput objects with actual data for the prediction response.
        Uses the shape and type specifications from infer_outputs but populates with actual results.

        Args:
            current_results: Dictionary containing the execution results from workflow nodes
            payload: The original request payload to check which outputs are requested

        Returns:
            List of InferOutput objects with actual data matching the expected specifications
        """
        response_outputs = []

        # Get requested outputs from payload, or use all if none specified
        requested_output_names = []
        if infer_request.request_outputs:
            requested_output_names = [
                req_out.name for req_out in infer_request.request_outputs
            ]
        else:
            return response_outputs
            # requested_output_names = [out.name for out in cls.infer_outputs]

        for output_spec in infer_outputs:
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
            if node_id in node_results and node_results[node_id] is not None:
                result_data = get_value_at_index(
                    node_results[node_id], int(output_index)
                )

                # Convert the result data to the appropriate format for KServe
                converted_data = cls._convert_comfy_output_to_kserve(
                    result_data, output_type, output_spec.shape
                )

                # Create new InferOutput with actual data
                response_output = InferOutput(
                    name=output_spec.name,
                    shape=(
                        list(converted_data.shape)
                        if hasattr(converted_data, "shape")
                        else output_spec.shape
                    ),
                    datatype=output_spec.datatype,
                    parameters=output_spec.parameters,
                )

                # Set the actual data
                # response_output.data = converted_data.tolist() if hasattr(converted_data, 'tolist') else [converted_data]
                response_output.data = converted_data
                response_outputs.append(response_output)

        return response_outputs

    @staticmethod
    def _convert_comfy_output_to_kserve(data: Any, output_type: str, expected_shape: List[int]
    ) -> Any:
        # TODO check it can be written shorter

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
            if hasattr(obj, "is_cuda") and obj.is_cuda:  # PyTorch tensor on GPU
                return obj.detach().to("cpu", non_blocking=True).contiguous()
            elif hasattr(obj, "cpu"):  # Any tensor with .cpu() method
                return obj.cpu().detach() if hasattr(obj, "detach") else obj.cpu()
            elif isinstance(obj, (list, tuple)):
                return type(obj)(to_cpu(x) for x in obj)
            elif isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            return obj

        # Convert input data to CPU first - this is the key addition
        data = to_cpu(data)

        # TODO replace strings with enums
        # Handle different output types (same logic as before, but now working with CPU data)
        if output_type in ["IMAGE", "MASK", "LATENT", "VIDEO", "NOISE"]:
            # Convert tensor-like data to numpy arrays
            if hasattr(data, "cpu"):  # PyTorch tensor (should now be on CPU)
                return data.cpu().numpy()
            elif hasattr(data, "numpy"):  # Other tensor types
                return data.numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                return np.array(data)

        elif output_type in ["STRING"]:
            # Convert to bytes for BYTES datatype
            if isinstance(data, str):
                return data.encode("utf-8")
            return str(data).encode("utf-8")

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

