from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic.dataclasses import dataclass

from models.model import Model, ModelName, PromptingState

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenAIPromptParameters(PromptingState):
    model: str = "gpt-4-0125-preview"
    n: int = 1
    presence_penalty: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    logprobs: bool = False
    top_logprobs: int | None = None


class OpenAIModel(Model):
    GPT_MODELS = (ModelName.GPT_4_TURBO, ModelName.GPT_4O)
    TOGETHER_AI_MODELS = (ModelName.LLAMA_3_70B_TOGETHER, ModelName.MIXTRAL_8X22B_TOGETHER)
    SUPPORTED_MODELS = (*GPT_MODELS, *TOGETHER_AI_MODELS)
    DEFAULT_PARAMETERS = OpenAIPromptParameters(
        prompt=None,
        model_name=ModelName.GPT_4_TURBO,
        max_output_tokens=2048,
    )

    def __init__(
        self,
        name: ModelName,
        max_output_tokens: int,
        *,
        n: int | None = None,
        presence_penalty: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        **_,
    ) -> None:
        super().__init__(name, max_output_tokens)
        if name not in OpenAIModel.SUPPORTED_MODELS:
            msg = f"{name} is not a valid OpenAI model"
            raise ValueError(msg)

        load_dotenv()

        self.client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        parameters = {
            "n": n,
            "presence_penalty": presence_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }
        parameters = {k: v for k, v in parameters.items() if v is not None}
        self._default_parameters = OpenAIPromptParameters(
            prompt=None,
            model_name=name,
            max_output_tokens=max_output_tokens,
            model=self.model_version,
            **parameters,
        )

    @property
    def default_parameters(self) -> PromptingState:
        return self._default_parameters

    @property
    def _api_key(self) -> str | None:
        if self.name in OpenAIModel.GPT_MODELS:
            return os.environ.get("OPENAI_API_KEY")
        if self.name in OpenAIModel.TOGETHER_AI_MODELS:
            return os.environ.get("TOGETHER_API_KEY")
        raise NotImplementedError

    @property
    def _base_url(self) -> str | None:
        if self.name in OpenAIModel.GPT_MODELS:
            return None
        if self.name in OpenAIModel.TOGETHER_AI_MODELS:
            return "https://api.together.xyz/v1"
        raise NotImplementedError

    @property
    def model_version(self) -> str:
        if self.name == ModelName.GPT_4_TURBO:
            return "gpt-4-0125-preview"
        if self.name == ModelName.GPT_4O:
            return "gpt-4o-2024-05-13"
        if self.name == ModelName.LLAMA_3_70B_TOGETHER:
            return "meta-llama/Llama-3-70b-chat-hf"
        if self.name == ModelName.MIXTRAL_8X22B_TOGETHER:
            return "mistralai/Mixtral-8x22B-Instruct-v0.1"
        raise NotImplementedError

    def _call_chat_api(self, prompting_state: OpenAIPromptParameters) -> ChatCompletion:
        if prompting_state.prompt is None:
            msg = "Prompt cannot be None"
            raise ValueError(msg)

        return self.client.chat.completions.create(
            max_tokens=prompting_state.max_output_tokens,
            model=prompting_state.model,
            messages=[{"role": "user", "content": prompting_state.prompt}],
            n=prompting_state.n,
            presence_penalty=prompting_state.presence_penalty,
            temperature=prompting_state.temperature,
            top_p=prompting_state.top_p,
            logprobs=prompting_state.logprobs,
            top_logprobs=prompting_state.top_logprobs,
        )

    def _get_prompting_state(self, prompt: str) -> OpenAIPromptParameters:
        prompt_params_dict = dataclasses.asdict(self.default_parameters)
        prompt_params_dict["prompt"] = prompt

        return OpenAIPromptParameters(**prompt_params_dict)

    def get_prompting_state(self, prompt: str) -> PromptingState:
        return self._get_prompting_state(prompt)

    def prompt(self, prompt: str) -> tuple[str, PromptingState]:
        prompting_state = self._get_prompting_state(prompt)

        response = self._call_chat_api(prompting_state)

        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning("Exceeded max tokens %d", self.max_output_tokens)
        elif choice.finish_reason not in ("stop", "eos"):
            msg = f"Unexpected finish reason {choice.finish_reason}"
            raise RuntimeError(msg)

        if choice.message.content is None:
            msg = f"No message returned by model {self.name} for prompt {prompt}"
            raise RuntimeError(msg)

        return choice.message.content, prompting_state

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> tuple[str, dict[str, float], PromptingState]:
        prompting_state = self._get_prompting_state(prompt)
        prompt_params_dict = dataclasses.asdict(prompting_state)
        prompt_params_dict["max_output_tokens"] = max_new_tokens
        prompt_params_dict["logprobs"] = True
        prompt_params_dict["top_logprobs"] = 8

        prompting_state = OpenAIPromptParameters(**prompt_params_dict)

        response = self._call_chat_api(prompting_state)

        choice = response.choices[0]
        if choice.finish_reason not in ("length", "stop"):
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

        return choice.message.content, token_probabilities, prompting_state
