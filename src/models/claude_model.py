from __future__ import annotations

import dataclasses
import logging
from typing import Any, Self

from dotenv import load_dotenv
from pydantic.dataclasses import dataclass

from models.model import Model, ModelName, PromptingState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaudePromptParameters(PromptingState):
    # TODO: Add anything that represents the state of the world when the model is being prompted
    pass


class ClaudeModel(Model):
    # Add all Claude models in this tuple. This shouldn't need to change.
    SUPPORTED_MODELS = (ModelName.CLAUDE_OPUS,)
    # Set some defaults. This shouldn't need to change.
    DEFAULT_PARAMETERS = ClaudePromptParameters(
        prompt=None, model_name=ModelName.CLAUDE_OPUS, max_output_tokens=2048,
    )

    def __init__(self, name: ModelName, max_output_tokens: int, **_) -> None:
        super().__init__(name, max_output_tokens)
        if name not in ClaudeModel.SUPPORTED_MODELS:
            msg = f"{name} is not a valid Claude model"
            raise ValueError(msg)

        load_dotenv()

        # TODO: Your init code, if desired.

    @classmethod
    def get_default_parameters(cls: type[Self]) -> PromptingState:
        return ClaudeModel.DEFAULT_PARAMETERS

    def _get_prompting_state(self, prompt: str) -> ClaudePromptParameters:
        prompt_params_dict = dataclasses.asdict(self.get_default_parameters())
        prompt_params_dict["prompt"] = prompt
        prompt_params_dict["name"] = self.name
        prompt_params_dict["max_output_tokens"] = self.max_output_tokens

        return ClaudePromptParameters(**prompt_params_dict)

    def get_prompting_state(self, prompt: str) -> PromptingState:
        return self._get_prompting_state(prompt)

    def prompt(self, prompt: str) -> tuple[str, PromptingState]:
        prompting_state = self._get_prompting_state(prompt)
        prompt_params_dict: dict[str, Any] = dataclasses.asdict(prompting_state)

        # TODO: Run the model. If you want to store any info capturing global state
        # you can put it in `prompt_params_dict` dictionary.
        # `self.max_output_tokens` contains the max number of output tokens.
        output = ""

        return output, ClaudePromptParameters(**prompt_params_dict)

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 5,
    ) -> tuple[str, dict[str, float], PromptingState]:
        prompting_state = self._get_prompting_state(prompt)
        prompt_params_dict: dict[str, Any] = dataclasses.asdict(prompting_state)
        prompt_params_dict["max_output_tokens"] = max_new_tokens

        # TODO: Run the model, and also get probability distribution over the first token.
        # You DO NOT need this for now (this will be used for MC later).
        # If you want to store any info capturing global state
        # you can put it in `prompt_params_dict` dictionary.
        output = ""
        token_probabilities: dict[str, float] = {}

        return output, token_probabilities, ClaudePromptParameters(**prompt_params_dict)
