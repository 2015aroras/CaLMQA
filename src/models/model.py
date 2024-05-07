from __future__ import annotations

import enum
from abc import ABCMeta, abstractmethod
from typing import Any, Self

from pydantic import Field as PyField
from pydantic.dataclasses import dataclass

__all__ = [
    "Model",
]


class ModelName(enum.Enum):
    HUMAN = "Human"
    HUMAN_DOT_POINTS = "Human (dot points)"
    GPT_4 = "GPT 4"
    AYA_101 = "AYA 13B"
    GEMMA_7B = "Gemma 7B"
    MIXTRAL_8X7B = "Mixtral 8x7B"
    MIXTRAL_8X22B_API = "Mixtral 8x22B (API)"
    MIXTRAL_8X22B_TOGETHER = "Mixtral 8x22B (together.ai)"
    XGLM_7_5B = "XGLM 7.5B"
    CLAUDE_OPUS = "Claude Opus"


@dataclass(frozen=True)
class PromptingState:
    """State when a model is being prompted.

    Holds information that (ideally) can represent the global state when
    a model is being prompted, so that the model prompting can be reproduced.
    """

    prompt: str | None
    model_name: ModelName
    max_output_tokens: int
    other_state: dict = PyField(default_factory=dict)

    @classmethod
    def make(cls: type[Self], **kwargs) -> PromptingState:
        model_name = kwargs["model_name"]
        assert isinstance(model_name, ModelName)

        from models.claude_model import ClaudeModel, ClaudePromptParameters
        from models.mistral_model import MistralModel, MistralPromptingState
        from models.openai_model import OpenAIModel, OpenAIPromptParameters
        from models.transformers_model import TransformersModel, TransformersPromptParameters

        if model_name in MistralModel.SUPPORTED_MODELS:
            return MistralPromptingState(**kwargs)
        if model_name in OpenAIModel.SUPPORTED_MODELS:
            return OpenAIPromptParameters(**kwargs)
        if model_name in TransformersModel.SUPPORTED_MODELS:
            return TransformersPromptParameters(**kwargs)
        if model_name in ClaudeModel.SUPPORTED_MODELS:
            return ClaudePromptParameters(**kwargs)

        raise NotImplementedError

    @staticmethod
    def get_discriminator_value(v: Any) -> str:
        from models.claude_model import ClaudeModel, ClaudePromptParameters
        from models.mistral_model import MistralModel, MistralPromptingState
        from models.openai_model import OpenAIModel, OpenAIPromptParameters
        from models.transformers_model import TransformersModel, TransformersPromptParameters

        model_name = ModelName(v["model_name"]) if isinstance(v, dict) else v.model_name
        if model_name in MistralModel.SUPPORTED_MODELS:
            return MistralPromptingState.__name__
        if model_name in OpenAIModel.SUPPORTED_MODELS:
            return OpenAIPromptParameters.__name__
        if model_name in TransformersModel.SUPPORTED_MODELS:
            return TransformersPromptParameters.__name__
        if model_name in ClaudeModel.SUPPORTED_MODELS:
            return ClaudePromptParameters.__name__
        if model_name in (ModelName.HUMAN, ModelName.HUMAN_DOT_POINTS):
            return PromptingState.__name__

        raise NotImplementedError


class Model(metaclass=ABCMeta):
    def __init__(self, name: ModelName, max_output_tokens: int) -> None:
        self.name = name
        self.max_output_tokens = max_output_tokens

    @property
    @abstractmethod
    def default_parameters(self) -> PromptingState:
        pass

    @abstractmethod
    def get_prompting_state(self, prompt: str) -> PromptingState:
        pass

    @abstractmethod
    def prompt(self, prompt: str) -> tuple[str, PromptingState]:
        pass

    @abstractmethod
    def prompt_and_next_token_probs(
        self, prompt: str, max_new_tokens: int = 128,
    ) -> tuple[str, dict[str, float], PromptingState]:
        """Prompts the model and retrieves the probabilities for the first generated token."""

    @classmethod
    def make(cls: type[Self], model_name: ModelName, *args, **kwargs) -> Model:
        from models.claude_model import ClaudeModel
        from models.mistral_model import MistralModel
        from models.openai_model import OpenAIModel
        from models.transformers_model import TransformersModel

        if model_name in ClaudeModel.SUPPORTED_MODELS:
            return ClaudeModel(model_name, *args, **kwargs)
        if model_name in MistralModel.SUPPORTED_MODELS:
            return MistralModel(model_name, *args, **kwargs)
        if model_name in OpenAIModel.SUPPORTED_MODELS:
            return OpenAIModel(model_name, *args, **kwargs)
        if model_name in TransformersModel.SUPPORTED_MODELS:
            return TransformersModel.make(model_name, *args, **kwargs)

        raise NotImplementedError
