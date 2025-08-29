import argparse
import kserve
import asyncio
import sys
from kserve import logging
from kserve.errors import ModelMissingError
from kserve.logging import logger
from .kutils import FileHandler


# TODO add WF as annotiation
parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument(
    "--workflow",
    type=str,
    required=True,
    help="A local path to the workflow",
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


async def init_extra_nodes(
    intit_custom_nodes: bool = False, init_api_nodes: bool = False
):
    from nodes import init_extra_nodes
    await init_extra_nodes(intit_custom_nodes, init_api_nodes)


if __name__ == "__main__":
    # Init them and get their path from args
    if args.configure_logging:
        logging.configure_logging(args.log_config_file)
        
    # TODO fix this cause the pyton path is also set
    sys.path.insert(0, args.comfyui_path)
    
    from nodes import NODE_CLASS_MAPPINGS

    from .kutils import load_extra_path_config

    # TODO error handling
    # TODO add extra flags for custom nodes paths or handle them here
    load_extra_path_config(args.models_path_config)

    if args.override_prompt_server:
        from .kutils import DummyObject
        import server
        server.PromptServer = DummyObject

    # Initialize the event loop for loading builtin extra nodes
    if args.enable_extra_builtin_nodes:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            init_extra_nodes(
                intit_custom_nodes=args.enable_custom_nodes,
                init_api_nodes=args.enable_api_nodes,
            )
        )

    if args.disable_save_nodes:
        disable_save_nodes(NODE_CLASS_MAPPINGS)

    if args.override_load_image_nodes:
        override_load_image_nodes(NODE_CLASS_MAPPINGS)
    
    

    workflow = FileHandler.read_json_file(args.workflow)
    from comfyserver import ComfyModel

    model = ComfyModel(args.model_name, workflow)
    
    try:
        # model.load()
        kserve.ModelServer().start([model])

    except ModelMissingError:
        logger.error(
            f"failed to locate workflow file for model {args.model_name} under dir {args.workflow},"
            f"trying loading from model repository."
        )
        # TODO Model mesh section
        # Case 1: Model will be loaded from model repository automatically, if present
        # Case 2: In the event that the model repository is empty, it's possible that this is a scenario for
        # multi-model serving. In such a case, models are loaded dynamically using the TrainedModel.
        # Therefore, we start the server without any preloaded models
        # kserve.ModelServer(
        #     registered_models=SKLearnModelRepository(args.model_dir)
        # ).start([])
