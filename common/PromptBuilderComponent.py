import os.path
import re
import yaml
import importlib
from abc import abstractmethod
from typing import Any, Dict, Type
from pydantic import BaseModel
from common import CallableComponent
from config.loader import settings, project_root

def camel_to_snake(name: str) -> str:
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

class PromptBuilderComponent(CallableComponent):
    """
    Base class for prompt builders. Expects each YAML to define:
      - system_prompt
      - user_prompt
      - assistant_prompt
      - user_postfix
      - pydantic_model

    Subclasses **must** override build_user() and build_assistant().
    Provides:
      - build_system()
      - build_user() [abstract]
      - build_assistant() [abstract]
      - build_postfix()
      - __call__()
      - get_schema_class()
      - get_response_parser()
      - parser (cached via @property)
    """
    def __init__(self):
        self._parser = None
        tpl_rel = settings["prompt_builder"]["templates_dir"]
        self.templates_dir = os.path.join(project_root, tpl_rel)

        cls_core = self.__class__.__name__.replace("PromptBuilder", "")
        key = getattr(self, "template_key", camel_to_snake(cls_core))
        filename = f"prompt_{key}.yml"
        path = os.path.join(self.templates_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prompt template not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            doc = yaml.safe_load(f)

        # Validate required sections
        for field in ("system_prompt", "user_prompt", "assistant_prompt", "user_postfix", "pydantic_model"):
            if field not in doc:
                raise KeyError(f"Prompt template missing required field: {field!r}")

        self.system_template  = doc["system_prompt"]
        self.user_template    = doc["user_prompt"]
        self.postfix_template = doc["user_postfix"]
        self.schema_path      = doc["pydantic_model"]
        self.assistant_template = doc["assistant_prompt"]
        self._tpl = doc

    def build_system(self, **context: Any) -> str:
        return self.system_template.format(**context)

    @abstractmethod
    def build_user(self, **context: Any) -> str:
        raise NotImplementedError("Subclasses must implement build_user()")

    @abstractmethod
    def build_assistant(self, **context: Any) -> str:
        """
        Render the assistant_prompt body.
        """
        raise NotImplementedError("Subclasses must implement build_assistant()")

    def build_postfix(self, **context: Any) -> str:
        """
        Render the user_postfix, injecting the JSON schema from the Pydantic model.
        Subclasses can override and call super().build_postfix(**context) to include the schema.
        """
        module_name, cls_name = self.schema_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_cls: Type[BaseModel] = getattr(module, cls_name)
        schema_json = model_cls.model_json_schema()
        return self.postfix_template.format(schema=schema_json, **context)

    def __call__(self, **context: Any) -> Dict[str, str]:
        return {
            "system":    self.build_system(**context) + self.build_postfix(**context),
            "user":      self.build_user(**context),
            "assistant": self.build_assistant(**context),
        }

    def get_schema_class(self) -> Type[BaseModel]:
        module, cls = self.schema_path.rsplit(".", 1)
        mod = importlib.import_module(module)
        return getattr(mod, cls)

    # ... existing parser property and related methods ...
    def get_response_parser(self) -> CallableComponent:
        module_name, cls_name = self.schema_path.rsplit(".", 1)
        parser_mod = importlib.import_module(module_name)
        parser_cls = getattr(parser_mod, cls_name + "Parser")
        return parser_cls()

    @property
    def parser(self) -> CallableComponent:
        if self._parser is None:
            self._parser = self.get_response_parser()
        return self._parser.parse