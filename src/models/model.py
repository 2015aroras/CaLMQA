from __future__ import annotations

import enum
from abc import ABCMeta, abstractmethod
from typing import Self

__all__ = [
    "Model",
]


class ModelName(enum.Enum):
    GPT_4 = enum.auto()
    AYA_101 = enum.auto()
    GEMMA_7B = enum.auto()
    MIXTRAL_8X7B = enum.auto()
    XGLM_7_5B = enum.auto()


class Model(metaclass=ABCMeta):
    def __init__(self, name: ModelName) -> None:
        self.name = name

    @abstractmethod
    def prompt(self, prompt: str) -> str:
        pass

    @classmethod
    def make(cls: type[Self], model_name: ModelName) -> Model:
        from models.openai_model import OpenAIModel
        from models.transformers_model import TransformersModel

        if model_name in OpenAIModel.SUPPORTED_MODELS:
            return OpenAIModel(model_name)
        if model_name in TransformersModel.SUPPORTED_MODELS:
            return TransformersModel(model_name)

        raise NotImplementedError
