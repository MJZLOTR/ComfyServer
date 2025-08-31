import argparse
import asyncio
import sys

import kserve
from kserve import logging
from kserve.errors import ModelMissingError
from kserve.logging import logger

from .kutils import FileHandler, load_extra_path_config
from .model import ComfyModel
from .model_repository import ConfyModelRepository


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument(
    "--workflow",
    type=str,
    required=True,
    help="A local path to the workflow(s) API json file(s)",
)
parser.add_argument(
    "--comfyui_path",
    type=str,
    required=True,
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
    from .overridden_nodes import LoadImage, LoadImageMask
    node_class_mapping["LoadImage"] = LoadImage
    node_class_mapping["LoadImageMask"] = LoadImageMask


if __name__ == "__main__":
    # Init them and get their path from args
    if args.configure_logging:
        logging.configure_logging(args.log_config_file)
        
    # TODO fix this cause the pyton path is also set
    sys.path.insert(0, args.comfyui_path)
    from nodes import NODE_CLASS_MAPPINGS

    # TODO error handling
    # TODO add extra flags for custom nodes paths or handle them here
    load_extra_path_config(args.models_path_config)

    if args.override_prompt_server:
        import server
        from .kutils import DummyObject
        server.PromptServer = DummyObject

    # Initialize the event loop for loading builtin extra nodes
    if args.enable_extra_builtin_nodes:
        from nodes import init_extra_nodes
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            init_extra_nodes(
                init_custom_nodes=args.enable_custom_nodes,
                init_api_nodes=args.enable_api_nodes,
            )
        )

    if args.disable_save_nodes:
        disable_save_nodes(NODE_CLASS_MAPPINGS)

    if args.override_load_image_nodes:
        override_load_image_nodes(NODE_CLASS_MAPPINGS)
    
    
    try:
        # Check if workflow is a file or a directory
        if args.workflow.endswith(".json"):
            workflow = FileHandler.read_json_file(args.workflow)
        else:
            raise ModelMissingError(args.workflow)
            
        model = ComfyModel(args.model_name, workflow)
        model.load()
        kserve.ModelServer().start([model])


    except ModelMissingError:
        logger.info(
            f"No workflow .json file for model {args.model_name} under dir {args.workflow} detected,"
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
        logger.error(f"Failed to start the model server: {str(e)}")
        raise e