from __future__ import annotations

import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic.dataclasses import dataclass

from models.model import Model, ModelName, PromptingState

if TYPE_CHECKING:
    from anthropic.types import Message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaudePromptParameters(PromptingState):
    model: str = "claude-3-opus-20240229"
    temperature: float = 1.0


class ClaudeModel(Model):
    # Add all Claude models in this tuple. This shouldn't need to change.
    SUPPORTED_MODELS = (ModelName.CLAUDE_OPUS, ModelName.CLAUDE_3_5_SONNET)
    # Set some defaults. This shouldn't need to change.
    DEFAULT_PARAMETERS = ClaudePromptParameters(
        prompt=None,
        model_name=ModelName.CLAUDE_OPUS,
        max_output_tokens=2048,
    )

    def __init__(
        self, name: ModelName, max_output_tokens: int, temperature: float | None = None, **_,
    ) -> None:
        super().__init__(name, max_output_tokens)
        if name not in ClaudeModel.SUPPORTED_MODELS:
            msg = f"{name} is not a valid Claude model"
            raise ValueError(msg)

        load_dotenv()
        self.client = Anthropic()
        parameters = {
            "temperature": temperature,
        }
        parameters: dict[str, Any] = {k: v for k, v in parameters.items() if v is not None}
        self._default_parameters = ClaudePromptParameters(
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
    def model_version(self) -> str:
        if self.name == ModelName.CLAUDE_OPUS:
            return "claude-3-opus-20240229"
        if self.name == ModelName.CLAUDE_3_5_SONNET:
            return "claude-3-5-sonnet-20240620"
        raise NotImplementedError

    def _get_prompting_state(self, prompt: str) -> ClaudePromptParameters:
        prompt_params_dict = dataclasses.asdict(self.default_parameters)
        prompt_params_dict["prompt"] = prompt

        return ClaudePromptParameters(**prompt_params_dict)

    def get_prompting_state(self, prompt: str) -> PromptingState:
        return self._get_prompting_state(prompt)

    def _call_messages_api(self, prompting_state: ClaudePromptParameters) -> Message:
        if prompting_state.prompt is None:
            msg = "Prompt cannot be None"
            raise ValueError(msg)

        max_attempts = 5
        for attempt_num in range(1, max_attempts + 1):
            try:
                return self.client.messages.create(
                    model=prompting_state.model,
                    messages=[{"role": "user", "content": prompting_state.prompt}],
                    temperature=prompting_state.temperature,
                    max_tokens=prompting_state.max_output_tokens,
                )
            except RuntimeError:  # noqa: PERF203
                wait_time = np.power(2, attempt_num)
                logger.exception(
                    "Claude call failed on attempt %d. Waiting %d sec",
                    attempt_num,
                    wait_time,
                )
                time.sleep(wait_time)

        msg = f"Failed to call messages api {max_attempts} times"
        raise RuntimeError(msg)

    def prompt(self, prompt: str) -> tuple[str, PromptingState]:
        prompting_state = self._get_prompting_state(prompt)
        prompt_params_dict: dict[str, Any] = dataclasses.asdict(prompting_state)

        response = self._call_messages_api(prompting_state)

        if response.stop_reason == "max_tokens":
            logger.warning("Exceeded max tokens %d", self.max_output_tokens)
        elif response.stop_reason != "end_turn":
            msg = f"Unhandled stop reason {response.stop_reason}"
            raise RuntimeError(msg)

        if len(response.content) == 0:
            msg = f"No message returned by model {self.name} for prompt {prompt}"
            raise RuntimeError(msg)

        output = response.content[0].text
        return output, ClaudePromptParameters(**prompt_params_dict)

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> tuple[str, dict[str, float], PromptingState]:
        raise NotImplementedError
