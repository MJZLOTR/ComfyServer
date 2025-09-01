import asyncio
import gc
import inspect
import random
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from kserve import Model
from kserve.errors import InvalidInput
from kserve.logging import logger, trace_logger
from kserve.protocol.infer_type import InferRequest, InferResponse
from nodes import NODE_CLASS_MAPPINGS

from .adaptor import ComfyKserveMapper
from .kutils import LoadOrderDeterminer, get_value_at_index


class ComfyModel(Model):
    """Executes a workflow directly instead of generating code."""

    def __init__(self, name: str, workflow: Dict):
        super().__init__(name)
        self.name = name
        # TODO add metadata dict to workflow and extract it herer
        self.workflow = workflow
        self.ready = False

        self._loader_outputs: dict[str, Any] = {}  # <- cached results
        self.gpu_semaphore = asyncio.Semaphore()

        
        logger.info(f"Initializing Comfy Model {name}")

        load_order_determiner = LoadOrderDeterminer(workflow, NODE_CLASS_MAPPINGS)
        self.orders = load_order_determiner.determine_load_order()

        # Map workflow to inference inputs and outputs
        self.infer_inputs, self.infer_outputs = (
            ComfyKserveMapper.convert_workflow_to_inference_objects(self.workflow)
        )

        # Save input and output names for quick validation
        self.input_names = [inp.name for inp in self.infer_inputs]
        self.output_names = [out.name for out in self.infer_outputs]
        logger.debug(f"{name} initialized with {self.input_names} as inputs and {self.output_names} as outputs")
        
    @staticmethod
    def _grabage_collect():
        # TODO: research if this is really necessary here or if it can be done better
        logger.debug("Grabage cleening")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            logger.debug("GPU memory cleared")

    async def get_input_types(self) -> List[Dict]:
        logger.debug(f"Getting input types for {self.name}")
        return [input_inf.to_dict() for input_inf in self.infer_inputs]

    async def get_output_types(self) -> List[Dict]:
        logger.debug(f"Getting output types for {self.name}")
        return [output_inf.to_dict() for output_inf in self.infer_outputs]

    def stop(self):
        logger.info(f"Stopping {self.name}")
        super().stop()
        self._loader_outputs = {}
        self._grabage_collect()

    def load(self) -> bool:
        "Execute all loader nodes exactly once and cache their results."

        initialized_objects = {}
        cache_dict = {}
        
        init_order, _, _ = self.orders
        
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
                    logger.debug(f"Instantiated node class: {cls_name}")

                obj = initialized_objects[cls_name]

                # TODO rewrite the caching logic to be more readable
                output = None
                for node_id, node_config in cache_dict.items():
                    if node_id == idx:
                        continue
                    if (
                        node_config["class_type"] == cls_name
                        and node_config["inputs"] == inputs
                    ):
                        logger.debug(f"Reusing loader node {node_id} ({cls_name}) for node {idx} with same inputs {inputs}")
                        output = self._loader_outputs[node_id]
                        break

                if not output:
                    logger.debug(f"Executing loader node {idx} ({cls_name}) with inputs {inputs}")
                    output = getattr(obj, obj.FUNCTION)(**inputs)

                # run and cache
                self._loader_outputs[idx] = output
               
                
        self.ready = True
        logger.info(f"{len(self._loader_outputs)} Loader nodes created for {self.name}")
        return self.ready

    async def preprocess(
        self, payload: InferRequest, headers: Dict[str, str] = None
    ) -> InferRequest:
        trace_logger.info(f"Preprocessing request for model {self.name}")
        
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
        trace_logger.info(f"Postprocessing request for model {self.name}")

        # TODO inplement output name relabeling here based on parameters in InferResponse comming from InferRequest in predict
        return result

    async def predict(
        self, payload: InferRequest, headers: Dict[str, str] = None
    ) -> Union[InferResponse, asyncio.Task]:
        trace_logger.info(f"Prediction request for model {self.name}")
        
        async with self.gpu_semaphore:
            try:
                return await asyncio.to_thread(
                    self._infer, payload=payload, headers=headers
                )
            except Exception as e:
                trace_logger.error(f"Inference request {payload} with headers {headers} for model {self.name} failed:  {str(e)}")
                raise e
            finally:
                self._grabage_collect()
    def _infer(
        self, payload: InferRequest, headers: Dict[str, str] = None
    ) -> InferResponse:
        """Execute the workflow directly and return results."""
        trace_logger.debug("Executing workflow")
        
        _, runtime_order, _ = self.orders

        # Store initialized objects and executed variables
        initialized_objects = {}
        current_results = {}
        
        device = torch.cuda.current_device()
        with torch.cuda.device(device):
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
                        trace_logger.warning(f"Skipping node {idx} ({class_type}) - missing required inputs")
                        continue

                    # Skip preview image nodes
                    if class_type == "PreviewImage":
                        trace_logger.debug(f"Skipping PreviewImage node {idx}")
                        continue

                    # Initialize the class if not already done
                    if class_type not in initialized_objects:
                        initialized_objects[class_type] = NODE_CLASS_MAPPINGS[class_type]()
                        trace_logger.debug(f"Initialized {class_type}")
                        
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
                                trace_logger.warning(f"Referenced node {node_id} not found in executed variables")
                                processed_inputs[key] = value
                                
                        elif user_input := payload.get_input_by_name(
                            ComfyKserveMapper.INFER_INPUT_NAME_TEMPLATE.format(
                                node_id=idx, input_name=key
                            )
                        ):

                            processed_inputs[key] = (
                                ComfyKserveMapper.convert_inferInput_to_comfy(
                                    user_input
                                )
                            )
                            trace_logger.debug(f"Using user input for {key} in node {idx}")

                        elif key in ["noise_seed", "seed"]:
                            # Generate random seed
                            processed_inputs[key] = random.randint(1, 2**64)
                            trace_logger.debug(f"Generated random seed for {key}: {processed_inputs[key]}")
                            
                        else:
                            processed_inputs[key] = value


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
                        trace_logger.debug(f"Executed node {idx} ({class_type}) successfully")

                    except Exception as e:
                        trace_logger.error(f"Error executing node {idx} ({class_type}): {str(e)}")
                        current_results[idx] = None

            # Create response outputs with actual data
            response_outputs = ComfyKserveMapper.generate_inference_response_outputs(
                node_results=current_results,
                infer_request=payload,
                infer_outputs=self.infer_outputs,
            )

        return InferResponse(
            response_id=str(payload.id),
            model_name=self.name,
            infer_outputs=response_outputs,
            parameters={"execution_successful": True},
        )

    def _process_inputs(self, inputs: Dict, executed_variables: Dict) -> Dict:
        """Process inputs, replacing references with actual values."""
        # TODO this is used just for loaders nodes
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
                    logger.warning(f"Referenced node {node_id} not found in executed variables")
                    processed_inputs[key] = value
            # TODO insert input replacement logic here
            elif key in ["noise_seed", "seed"]:
                # Generate random seed
                processed_inputs[key] = random.randint(1, 2**64)
            else:
                processed_inputs[key] = value

        return processed_inputs

    def _get_function_parameters(self, func: Callable) -> List | None:
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
        except Exception as e:
            logger.error(f"Could not extract function parameters: {str(e)}")
            return None
