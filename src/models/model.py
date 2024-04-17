from __future__ import annotations

import enum
from abc import ABCMeta, abstractmethod
from typing import Self

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

__all__ = [
    "Model",
]


class ModelName(enum.Enum):
    HUMAN = "Human"
    GPT_4 = "GPT 4"
    AYA_101 = "AYA 13B"
    GEMMA_7B = "Gemma 7B"
    MIXTRAL_8X7B = "Mixtral 8x7B"
    XGLM_7_5B = "XGLM 7.5B"


@dataclass(frozen=True, config=ConfigDict(extra="allow"))
class PromptingState:
    """State when a model is being prompted.

    Holds information that (ideally) can represent the global state when
    a model is being prompted, so that the model prompting can be reproduced.
    """

    prompt: str | None
    model_name: ModelName
    max_output_tokens: int

    @classmethod
    def make(cls: type[Self], **kwargs) -> PromptingState:
        model_name = kwargs["model_name"]
        assert isinstance(model_name, ModelName)

        from models.openai_model import OpenAIModel, OpenAIPromptParameters
        from models.transformers_model import TransformersModel, TransformersPromptParameters

        if model_name in OpenAIModel.SUPPORTED_MODELS:
            return OpenAIPromptParameters(**kwargs)
        if model_name in TransformersModel.SUPPORTED_MODELS:
            return TransformersPromptParameters(**kwargs)

        raise NotImplementedError


class Model(metaclass=ABCMeta):
    def __init__(self, name: ModelName, max_output_tokens: int) -> None:
        self.name = name
        self.max_output_tokens = max_output_tokens

    @classmethod
    @abstractmethod
    def get_default_parameters(cls: type[Self]) -> PromptingState:
        pass

    @abstractmethod
    def get_prompting_state(self, prompt: str) -> PromptingState:
        pass

    @abstractmethod
    def prompt(self, prompt: str) -> tuple[str, PromptingState]:
        pass

    @abstractmethod
    def prompt_and_next_token_probs(
        self, prompt: str, max_new_tokens: int = 5,
    ) -> tuple[str, dict[str, float], PromptingState]:
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
