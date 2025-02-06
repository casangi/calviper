
def register_subclass(cls):
    """
    There will be a lot more added here bt for testing .....
    """
    import importlib

    if cls.__name__ == "GainMatrixDataSet":
        importlib.import_module("calviper.factory.gain")

    return cls