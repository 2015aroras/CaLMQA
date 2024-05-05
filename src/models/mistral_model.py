from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from pydantic.dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from models.model import Model, ModelName, PromptingState

if TYPE_CHECKING:
    from mistralai.models.chat_completion import ChatCompletionResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MistralPromptingState(PromptingState):
    model: str = "open-mixtral-8x22b-2404"
    n: int = 1
    temperature: float = 0.7
    top_p: float = 1.0
    transformers_model_path: str = "mistralai/Mixtral-8x22B-Instruct-v0.1"


class MistralModel(Model):
    SUPPORTED_MODELS = (ModelName.MIXTRAL_8X22B_API,)
    DEFAULT_PARAMETERS = MistralPromptingState(
        prompt=None,
        model_name=ModelName.MIXTRAL_8X22B_API,
        max_output_tokens=2048,
    )

    def __init__(
        self,
        name: ModelName,
        max_output_tokens: int,
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 1.0,
        transformers_model_path: str = "mistralai/Mixtral-8x22B-Instruct-v0.1",
        **_,
    ) -> None:
        super().__init__(name, max_output_tokens)
        if name not in MistralModel.SUPPORTED_MODELS:
            msg = f"{name} is not a valid Mistral model"
            raise ValueError(msg)

        load_dotenv()
        api_key = os.environ["MISTRAL_API_KEY"]
        self.hf_token = os.environ.get("HF_USER_ACCESS_TOKEN")

        self.client = MistralClient(api_key=api_key)
        self.tokenizer = self._init_tokenizer()
        self._default_parameters = MistralPromptingState(
            prompt=None,
            model_name=name,
            max_output_tokens=max_output_tokens,
            model=self.model_version,
            n=n,
            temperature=temperature,
            top_p=top_p,
            transformers_model_path=transformers_model_path,
        )

    @property
    def default_parameters(self) -> PromptingState:
        return self._default_parameters

    def _init_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        default_parameters = self.default_parameters
        assert isinstance(default_parameters, MistralPromptingState)
        assert default_parameters.transformers_model_path is not None

        return AutoTokenizer.from_pretrained(
            default_parameters.transformers_model_path,
            token=self.hf_token,
        )

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
            messages=[ChatMessage(role="user", content=prompting_state.prompt)],
        )

    def _get_prompting_state(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> MistralPromptingState:
        prompt_params_dict = dataclasses.asdict(self.default_parameters)
        prompt_params_dict["prompt"] = prompt
        prompt_params_dict["max_output_tokens"] = max_new_tokens or self.max_output_tokens

        return MistralPromptingState(**prompt_params_dict)

    def get_prompting_state(self, prompt: str) -> PromptingState:
        return self._get_prompting_state(prompt)

    def _prompt(self, prompt: str, max_new_tokens: int = 128) -> tuple[str, PromptingState]:
        prompting_state = self._get_prompting_state(prompt, max_new_tokens)

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

    def prompt(self, prompt: str) -> tuple[str, PromptingState]:
        return self._prompt(prompt, self.max_output_tokens)

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> tuple[str, dict[str, float], PromptingState]:
        output, prompting_state = self._prompt(prompt, max_new_tokens)
        tokenization = self.tokenizer.tokenize(output)
        first_token = tokenization[0]

        next_token_probs = {first_token: 1.0}
        return output, next_token_probs, prompting_state
