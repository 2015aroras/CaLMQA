from __future__ import annotations

import enum
from abc import ABCMeta, abstractmethod
from typing import Self

from pydantic.dataclasses import dataclass

__all__ = [
    "Model",
]


class ModelName(enum.Enum):
    HUMAN = enum.auto()
    GPT_4 = enum.auto()
    AYA_101 = enum.auto()
    GEMMA_7B = enum.auto()
    MIXTRAL_8X7B = enum.auto()
    XGLM_7_5B = enum.auto()


@dataclass(frozen=True)
class PromptParameters:
    """Model parameters when prompting.

    Holds parameters that (ideally) can represent the state of a model when
    it is being prompted.
    """

    prompt: str | None
    name: ModelName
    max_output_tokens: int


class Model(metaclass=ABCMeta):
    def __init__(self, name: ModelName, max_output_tokens: int) -> None:
        self.name = name
        self.max_output_tokens = max_output_tokens

    @classmethod
    @abstractmethod
    def get_default_parameters(cls: type[Self]) -> PromptParameters:
        pass

    @abstractmethod
    def prompt(self, prompt: str) -> tuple[str, PromptParameters]:
        pass

    @abstractmethod
    def prompt_and_next_token_probs(
        self, prompt: str, max_new_tokens: int = 5,
    ) -> tuple[str, dict[str, float], PromptParameters]:
        """Prompts the model and retrieves the probabilities for the first generated token."""

    @classmethod
    def make(cls: type[Self], model_name: ModelName, *args, **kwargs) -> Model:
        from models.openai_model import OpenAIModel
        from models.transformers_model import TransformersModel

        if model_name in OpenAIModel.SUPPORTED_MODELS:
            return OpenAIModel(model_name, *args, **kwargs)
        if model_name in TransformersModel.SUPPORTED_MODELS:
            return TransformersModel.make(model_name, *args, **kwargs)

        raise NotImplementedError
