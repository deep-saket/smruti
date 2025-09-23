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

    _registry = {}

    @classmethod
    def load_models(cls):
        models_pkg = importlib.import_module("models")
        instantiated = {}
        for attr_name, class_name in agent["models"].items():
            if class_name in instantiated:
                instance = instantiated[class_name]
            else:
                try:
                    klass = getattr(models_pkg, class_name)
                except AttributeError as e:
                    raise ImportError(f"Model class '{class_name}' not found in 'models' package : {e}")
                except Exception as e:
                    raise ImportError(f"Error importing model class '{class_name}': {e}") from e
                cfg = settings["models"].get(class_name, {})
                instance = klass(**cfg)
            # attach to the class so you can do ModelManager.<attr_name>
            setattr(cls, attr_name, instance)
            # also track in the registry for other modules that check it
            instantiated[class_name] = instance
            cls._registry[attr_name] = instance

        return cls
