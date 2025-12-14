import argparse
import asyncio
import sys
import os

import kserve
from kserve import logging
from kserve.errors import ModelMissingError

from .kutils import FileHandler, load_extra_path_config
from .model import ComfyModel
from .model_repository import ConfyModelRepository

logger = logging.logger

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument(
    "--workflow",
    type=str,
    required=True,
    help="A local path to the workflow(s) API json file(s)",
)
parser.add_argument(
    "--comfy_path",
    type=str,
    default=None,
    required=False,
    help="A local path to the ComfyUI directory",
)
parser.add_argument(
    # TODO if required then defatult is not needed
    "--models_path_config",
    type=str,
    # default="/mnt/models/models_path_config.yaml",
    required=True,
    help="Path to the models path config yaml file",
)
parser.add_argument(
    "--disable_save_nodes",
    type=bool,
    default=True,
    required=False,
    help="Disables nodes which have a saving functionality",
)
parser.add_argument(
    "--disable_progress_bar",
    type=bool,
    default=True,
    required=False,
    help="Disables progress bar when sampling",
)
parser.add_argument(
    "--override_load_image_nodes",
    type=bool,
    default=True,
    required=False,
    help="Overrides the LoadImage and LoadImageMask nodes to load images also from API calls",
)
parser.add_argument(
    "--override_prompt_server",
    type=bool,
    default=True,
    required=False,
    help="Overrides the PromptServer with dummy server to avoid issue with GetImageSize node",
)
parser.add_argument(
    "--enable_extra_builtin_nodes",
    type=bool,
    default=True,
    required=False,
    help="Loads and initializes extra builtin nodes(disable if not used)",
)
parser.add_argument(
    "--enable_api_nodes",
    type=bool,
    default=False,
    required=False,
    help="Loads and initializes 3rd party API nodes ",
)
parser.add_argument(
    "--enable_custom_nodes",
    type=bool,
    default=False,
    required=False,
    help="Loads and initializes custom nodes(they should be allready in custum_nodes folder)",
)
args, _ = parser.parse_known_args()


def disable_save_nodes(node_class_mapping: dict):
    """
    Disables nodes which have a saving functionality.
    :param flags: The flags to disable save nodes.
    """
    logger.info("Disabling save nodes")
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    def bypass_function(*args, **kwargs) -> None:
        return None

    for _, node_class in node_class_mapping.items():
        if node_class.FUNCTION.startswith("save") and node_class.OUTPUT_NODE:
            setattr(node_class, node_class.FUNCTION, bypass_function)
            setattr(node_class, "INPUT_TYPES", INPUT_TYPES)
            # setattr(node_class, "RETURN_TYPES", "IGNORE")

def override_load_image_nodes(node_class_mapping: dict):
    logger.info("Overriding load image nodes")
    from .overridden_nodes import LoadImage, LoadImageMask
    node_class_mapping["LoadImage"] = LoadImage
    node_class_mapping["LoadImageMask"] = LoadImageMask



# TODO change all log.error to log.exception
if __name__ == "__main__":
    if args.configure_logging:
        logging.configure_logging(args.log_config_file)
    
    if args.disable_progress_bar:
        import comfy.utils
        comfy.utils.PROGRESS_BAR_ENABLED=False
        
    # TODO fix this cause the pyton path is also set
    if args.comfy_path:
        comfy_path = args.comfy_path
    elif comfy_path:= os.environ.get("COMFY_PATH",""):
        pass
    else:
        RuntimeError(f"Path to Comfy is not detected either via env or arg") 
    logger.info(f"Adding ComfyUI path to sys.path: {comfy_path}")
    sys.path.insert(0, comfy_path)
    
    from nodes import NODE_CLASS_MAPPINGS

    # TODO error handling
    # TODO add extra flags for custom nodes paths or handle them here
    logger.info(f"Loading extra path config: {args.models_path_config}")
    load_extra_path_config(args.models_path_config)

    if args.override_prompt_server:
        logger.info("Overriding PromptServer with dummy implementation")
        import server

        from .kutils import DummyObject
        server.PromptServer = DummyObject

    # Initialize the event loop for loading builtin extra nodes
    if args.enable_extra_builtin_nodes:
        logger.debug("Initializing extra builtin nodes")
        from nodes import init_extra_nodes
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                init_extra_nodes(
                    init_custom_nodes=args.enable_custom_nodes,
                    init_api_nodes=args.enable_api_nodes,
                )
            )
            logger.info(f"Extra nodes initialized - Custom: {args.enable_custom_nodes}, API: {args.enable_api_nodes}")
        except Exception as e:
            logger.error(f"Failed to initialize extra nodes: {str(e)}")
            raise
        finally:
            loop.close()

    if args.disable_save_nodes:
        disable_save_nodes(NODE_CLASS_MAPPINGS)

    if args.override_load_image_nodes:
        override_load_image_nodes(NODE_CLASS_MAPPINGS)
    
    
    try:
        # Check if workflow is a file or a directory
        if args.workflow.endswith(".json"):
            logger.debug(f"Loading single workflow {args.workflow} as {args.model_name}")
            workflow = FileHandler.read_json_file(args.workflow)
            model = ComfyModel(args.model_name, workflow)
            
            if not model.load():
                logger.error(f"Failed to load workflow {workflow} as {args.model_name}")
                raise RuntimeError(f"Model loading failed")
                
            logger.info(f"Starting KServe server with single workflow {args.workflow} as {args.model_name}")
            kserve.ModelServer().start([model])
        else:
            raise ModelMissingError(args.workflow)


    except ModelMissingError:
        logger.info(
            f"No single workflow file for model {args.model_name} under dir {args.workflow} detected,"
            f"trying loading from model repository."
        )
        # TODO Model mesh section
        # Case 1: Model will be loaded from model repository automatically, if present
        # Case 2: In the event that the model repository is empty, it's possible that this is a scenario for
        # multi-model serving. In such a case, models are loaded dynamically using the TrainedModel.
        # Therefore, we start the server without any preloaded models
        # TODO read about TrainedModel
        kserve.ModelServer(
            registered_models=ConfyModelRepository(args.workflow)
        ).start([])
    except Exception as e:
        logger.exception(f"Failed to start the model server: {str(e)}")
        raise e