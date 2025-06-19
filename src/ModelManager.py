import importlib
from common import BaseComponent
from config.loader import settings

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
        for class_name in settings["agent"]["models"]:
            try:
                cls = getattr(models_pkg, class_name)
            except AttributeError:
                raise ImportError(f"Model class '{class_name}' not found in 'models' package")
            cfg = settings["models"].get(class_name, {})
            instance = cls(**cfg)
            # attach to the class so you can do ModelManager.<ClassName>
            setattr(ModelManager, class_name, instance)