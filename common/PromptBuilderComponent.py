# prompt/PromptBuilderComponent.py
import os.path
import re
import yaml
import importlib
from abc import abstractmethod
from typing import Any, Tuple, Type
from pydantic import BaseModel
from common import CallableComponent
from config.loader import settings, project_root

def camel_to_snake(name: str) -> str:
    """
    Convert CamelCase to snake_case.
    """
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

class PromptBuilderComponent(CallableComponent):
    """
    Base class for prompt builders. Expects each YAML to define:
      - system_prompt
      - user_prompt
      - user_postfix
      - pydantic_model

    Subclasses **must** override build_user().
    Provides:
      - build_system()
      - build_user() [abstract]
      - build_postfix()
      - get_schema_class()
      - get_response_parser()
      - parser (cached via @property)
    """
    def __init__(self):
        # Load templates_dir from settings
        tpl_rel = settings["prompt_builder"]["templates_dir"]
        self.templates_dir = os.path.join(project_root, tpl_rel)

        # Determine YAML filename from class name or override
        cls_core = self.__class__.__name__.replace("PromptBuilder", "")
        key = getattr(self, "template_key", camel_to_snake(cls_core))
        filename = f"prompt_{key}.yml"
        path = os.path.join(self.templates_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prompt template not found: {path}")

        # Load YAML and validate required sections
        with open(path, 'r', encoding='utf-8') as f:
            doc = yaml.safe_load(f)
            for section in ("system_prompt", "user_prompt", "user_postfix", "pydantic_model"):
                if section not in doc:
                    raise KeyError(f"'{section}' missing in {filename}")

        self.system_template  = doc["system_prompt"]
        self.user_template    = doc["user_prompt"]
        self.postfix_template = doc["user_postfix"]
        self.schema_path      = doc["pydantic_model"]

        # placeholder for lazy‐loaded parser
        self._parser = None

    def build_system(self, **context: Any) -> str:
        """Render the system_prompt section with provided context."""
        return self.system_template.format(**context)

    @abstractmethod
    def build_user(self, **context: Any) -> str:
        """Render the main user_prompt body. Must be implemented by subclasses."""
        raise NotImplementedError()

    def build_postfix(self, **context: Any) -> str:
        """
        Render the user_postfix, injecting the JSON schema from the Pydantic model.
        Subclasses can override and call super().build_postfix(**context) to include the schema.
        """
        module_name, cls_name = self.schema_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_cls: Type[BaseModel] = getattr(module, cls_name)
        schema_json = model_cls.schema_json(indent=2)
        return self.postfix_template.format(schema=schema_json, **context)

    def get_schema_class(self) -> Type[BaseModel]:
        """Dynamically import and return the Pydantic model specified in the YAML."""
        module_name, cls_name = self.schema_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        if not issubclass(cls, BaseModel):
            raise TypeError(f"{cls_name} is not a Pydantic BaseModel")
        return cls

    def get_response_parser(self) -> CallableComponent:
        """
        Dynamically import and return the corresponding parser class for this schema.
        E.g. 'schemas.lm.ChatResponse' → 'schemas.lm.ChatResponseParser'
        """
        module_name, cls_name = self.schema_path.rsplit(".", 1)
        parser_mod = importlib.import_module(module_name)
        parser_cls = getattr(parser_mod, cls_name + "Parser")
        return parser_cls()

    @property
    def parser(self) -> CallableComponent:
        """
        Lazily instantiate and cache the parser for this prompt's schema.
        Usage: resp = builder.parser(raw_json)
        """
        if self._parser is None:
            self._parser = self.get_response_parser()
        return self._parser

    def __call__(self, **context: Any) -> dict[str, str]:
        """
        Produce (system_prompt, full_user_prompt).
        full_user_prompt = build_user(**context) + newline + build_postfix(**context)
        """
        sys_txt  = self.build_system(**context) + "  " + self.build_postfix(**context)
        user_txt = self.build_user(**context)
        return {"system": sys_txt,
                "user": user_txt}



## /Users/saketm10/Projects/smruti/prompts/prompt_files/prompt_main.yml
## /Users/saketm10/Projects/smruti/prompts/prompt_files/prompt_main.yml