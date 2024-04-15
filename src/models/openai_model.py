from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Self

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic.dataclasses import dataclass

from models.model import Model, ModelName, PromptParameters

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenAIPromptParameters(PromptParameters):
    model: str = "gpt-4-0125-preview"
    n: int = 1
    presence_penalty: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    logprobs: bool = False
    top_logprobs: int | None = None


class OpenAIModel(Model):
    SUPPORTED_MODELS = (ModelName.GPT_4,)
    DEFAULT_PARAMETERS = OpenAIPromptParameters(
        prompt=None, name=ModelName.GPT_4, max_output_tokens=2048,
    )

    def __init__(self, name: ModelName, max_output_tokens: int, **_) -> None:
        super().__init__(name, max_output_tokens)
        if name not in OpenAIModel.SUPPORTED_MODELS:
            msg = f"{name} is not a valid OpenAI model"
            raise ValueError(msg)

        load_dotenv()

        self.client = OpenAI()

    @classmethod
    def get_default_parameters(cls: type[Self]) -> PromptParameters:
        return OpenAIModel.DEFAULT_PARAMETERS

    @property
    def model_version(self) -> str:
        if self.name == ModelName.GPT_4:
            return "gpt-4-0125-preview"
        raise NotImplementedError

    def _call_chat_api(self, prompt_parameters: OpenAIPromptParameters) -> ChatCompletion:
        if prompt_parameters.prompt is None:
            msg = "Prompt cannot be None"
            raise ValueError(msg)

        return self.client.chat.completions.create(
            max_tokens=prompt_parameters.max_output_tokens,
            model=prompt_parameters.model,
            messages=[{"role": "user", "content": prompt_parameters.prompt}],
            logprobs=prompt_parameters.logprobs,
            top_logprobs=prompt_parameters.top_logprobs,
        )

    def prompt(self, prompt: str) -> tuple[str, PromptParameters]:
        prompt_params_dict = dataclasses.asdict(self.get_default_parameters())
        prompt_params_dict["prompt"] = prompt
        prompt_params_dict["name"] = self.name
        prompt_params_dict["max_output_tokens"] = self.max_output_tokens
        prompt_params_dict["model"] = self.model_version

        prompt_parameters = OpenAIPromptParameters(**prompt_params_dict)

        response = self._call_chat_api(prompt_parameters)

        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning("Exceeded max tokens %d", self.max_output_tokens)
        elif choice.finish_reason != "stop":
            msg = f"Unexpected finish reason {choice.finish_reason}"
            raise RuntimeError(msg)

        if choice.message.content is None:
            msg = f"No message returned by model {self.name} for prompt {prompt}"
            raise RuntimeError(msg)

        return choice.message.content, prompt_parameters

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 5,
    ) -> tuple[str, dict[str, float], PromptParameters]:
        prompt_params_dict = dataclasses.asdict(self.get_default_parameters())
        prompt_params_dict["name"] = self.name
        prompt_params_dict["max_output_tokens"] = max_new_tokens
        prompt_params_dict["model"] = self.model_version
        prompt_params_dict["logprobs"] = True
        prompt_params_dict["top_logprobs"] = 8

        prompt_parameters = OpenAIPromptParameters(**prompt_params_dict)

        response = self._call_chat_api(prompt_parameters)

        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning("Exceeded max tokens %d", self.max_output_tokens)
        elif choice.finish_reason != "stop":
            msg = f"Unexpected finish reason {choice.finish_reason}"
            raise RuntimeError(msg)

        if choice.message.content is None:
            msg = f"No message returned by model {self.name} for prompt {prompt}"
            raise RuntimeError(msg)

        if choice.logprobs is None:
            msg = f"No logprobs returned for prompt {prompt}"
            raise RuntimeError(msg)
        if choice.logprobs.content is None or len(choice.logprobs.content) == 0:
            msg = f"No content in logprobs for prompt {prompt}"
            raise RuntimeError(msg)

        top_logprobs = choice.logprobs.content[0].top_logprobs
        token_probabilities = {}
        for top_logprob in top_logprobs:
            token_probabilities[top_logprob.token] = np.exp(top_logprob.logprob)

        return choice.message.content, token_probabilities, prompt_parameters
