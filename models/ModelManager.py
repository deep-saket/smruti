import importlib
from common import BaseComponent
from config.loader import settings, agent

class ModelManager(BaseComponent):
    """
    Dynamically loads and exposes model instances as class attributes.
    Model class names come from settings["agent"]["models"].
    Each model is imported from the `models` package and instantiated
    with settings["models"][<ClassName>] as kwargs.
    """
    _initialized = False

    def __init__(self):
        super().__init__()
        if not ModelManager._initialized:
            self._load_models()
            ModelManager._initialized = True

    def _load_models(self):
        models_pkg = importlib.import_module("models")
        instantiated = {}
        for attr_name, class_name in agent["models"].items():
            if class_name in instantiated:
                instance = instantiated[class_name]
            else:
                try:
                    cls = getattr(models_pkg, class_name)
                except AttributeError as e:
                    raise ImportError(f"Model class '{class_name}' not found in 'models' package : {e}")
                except Exception as e:
                    raise ImportError(f"Error importing model class '{class_name}': {e}") from e
                cfg = settings["models"].get(class_name, {})
                instance = cls(**cfg)
            # attach to the class so you can do ModelManager.<ClassName>
            setattr(ModelManager, attr_name, instance)
            instantiated[class_name] = instance