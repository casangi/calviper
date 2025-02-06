import pathlib
import json
import importlib

import toolviper.utils.logger as logger

def register_subclass(cls):
    """
    There will be a lot more added here bt for testing .....
    """

    config_path = str(pathlib.Path(__file__).parent.parent.joinpath("config").joinpath("subclasses.config.json").resolve())

    with open(config_path, "r") as file:
        object_ = json.load(file)
        if cls.__name__ in object_:
            importlib.import_module(object_[cls.__name__])

        else:
            logger.error(f"No subclass named {cls.__name__}")


    return cls

