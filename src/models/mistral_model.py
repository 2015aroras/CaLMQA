from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING, Self

from dotenv import load_dotenv
from mistralai.client import MistralClient
from pydantic.dataclasses import dataclass

from models.model import Model, ModelName, PromptingState

if TYPE_CHECKING:
    from mistralai.models.chat_completion import ChatCompletionResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MistralPromptingState(PromptingState):
    model: str = "open-mixtral-8x22b-2404"
    n: int = 1
    presence_penalty: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0


class MistralModel(Model):
    SUPPORTED_MODELS = (ModelName.MIXTRAL_8X22B_API,)
    DEFAULT_PARAMETERS = MistralPromptingState(
        prompt=None, model_name=ModelName.MIXTRAL_8X22B_API, max_output_tokens=2048,
    )

    def __init__(self, name: ModelName, max_output_tokens: int, **_) -> None:
        super().__init__(name, max_output_tokens)
        if name not in MistralModel.SUPPORTED_MODELS:
            msg = f"{name} is not a valid Mistral model"
            raise ValueError(msg)

        load_dotenv()
        api_key = os.environ["MISTRAL_API_KEY"]

        self.client = MistralClient(api_key=api_key)

    @classmethod
    def get_default_parameters(cls: type[Self]) -> PromptingState:
        return MistralModel.DEFAULT_PARAMETERS

    @property
    def model_version(self) -> str:
        if self.name == ModelName.MIXTRAL_8X22B_API:
            return "open-mixtral-8x22b-2404"
        raise NotImplementedError

    def _call_chat_api(self, prompting_state: MistralPromptingState) -> ChatCompletionResponse:
        if prompting_state.prompt is None:
            msg = "Prompt cannot be None"
            raise ValueError(msg)

        return self.client.chat(
            max_tokens=prompting_state.max_output_tokens,
            model=prompting_state.model,
            messages=[{"role": "user", "content": prompting_state.prompt}],
        )

    def _get_prompting_state(self, prompt: str) -> MistralPromptingState:
        prompt_params_dict = dataclasses.asdict(self.get_default_parameters())
        prompt_params_dict["prompt"] = prompt
        prompt_params_dict["name"] = self.name
        prompt_params_dict["max_output_tokens"] = self.max_output_tokens
        prompt_params_dict["model"] = self.model_version

        return MistralPromptingState(**prompt_params_dict)

    def get_prompting_state(self, prompt: str) -> PromptingState:
        return self._get_prompting_state(prompt)

    def prompt(self, prompt: str) -> tuple[str, PromptingState]:
        prompting_state = self._get_prompting_state(prompt)

        response = self._call_chat_api(prompting_state)

        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning("Exceeded max tokens %d", self.max_output_tokens)
        elif choice.finish_reason != "stop":
            msg = f"Unexpected finish reason {choice.finish_reason}"
            raise RuntimeError(msg)

        if choice.message.content is None:
            msg = f"No message returned by model {self.name} for prompt {prompt}"
            raise RuntimeError(msg)

        assert isinstance(choice.message.content, str)
        return choice.message.content, prompting_state

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> tuple[str, dict[str, float], PromptingState]:
        raise NotImplementedError
