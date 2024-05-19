from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any, Iterable

import numpy as np
import vertexai
from dotenv import load_dotenv
from pydantic.dataclasses import dataclass
from vertexai.preview.generative_models import FinishReason, GenerationResponse, GenerativeModel

from models.model import Model, ModelName, PromptingState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GooglePromptingState(PromptingState):
    model: str = "gemini-1.5-pro-preview-0514"
    temperature: float = 1.0


class GoogleModel(Model):
    # Add all Google models in this tuple. This shouldn't need to change.
    SUPPORTED_MODELS = (ModelName.GEMINI_1_5_PRO,)
    # Set some defaults. This shouldn't need to change.
    DEFAULT_PARAMETERS = GooglePromptingState(
        prompt=None,
        model_name=ModelName.GEMINI_1_5_PRO,
        max_output_tokens=2048,
    )

    def __init__(
        self,
        name: ModelName,
        max_output_tokens: int,
        temperature: float | None = None,
        **_,
    ) -> None:
        super().__init__(name, max_output_tokens)
        if name not in GoogleModel.SUPPORTED_MODELS:
            msg = f"{name} is not a valid Google model"
            raise ValueError(msg)

        load_dotenv()
        vertexai.init(
            project=os.environ["PROJECT_ID"],
            location=os.environ["REGION"],
        )
        self.model = GenerativeModel(self.model_version)
        parameters = {
            "temperature": temperature,
        }
        parameters: dict[str, Any] = {k: v for k, v in parameters.items() if v is not None}
        self._default_parameters = GooglePromptingState(
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
        if self.name == ModelName.GEMINI_1_5_PRO:
            return "gemini-1.5-pro-preview-0514"
        raise NotImplementedError

    def _get_prompting_state(self, prompt: str) -> GooglePromptingState:
        prompt_params_dict = dataclasses.asdict(self.default_parameters)
        prompt_params_dict["prompt"] = prompt

        return GooglePromptingState(**prompt_params_dict)

    def get_prompting_state(self, prompt: str) -> PromptingState:
        return self._get_prompting_state(prompt)

    def _call_generate_content_api(
        self,
        prompting_state: GooglePromptingState,
    ) -> GenerationResponse:
        if prompting_state.prompt is None:
            msg = "Prompt cannot be None"
            raise ValueError(msg)

        max_attempts = 5
        for attempt_num in range(1, max_attempts + 1):
            try:
                response = self.model.generate_content(
                    prompting_state.prompt,
                    generation_config={
                        "candidate_count": 1,
                        "max_output_tokens": prompting_state.max_output_tokens,
                        "temperature": prompting_state.temperature,
                    },
                )
                if isinstance(response, Iterable):
                    n_responses = len(list(response))
                    msg = f"Response is iterable of size {n_responses}, expected single response"
                    raise TypeError(msg)
                return response
            except RuntimeError:  # noqa: PERF203
                wait_time = np.power(2, attempt_num)
                logger.exception(
                    "Google call failed on attempt %d. Waiting %d sec",
                    attempt_num,
                    wait_time,
                )
                time.sleep(wait_time)

        msg = f"Failed to call messages api {max_attempts} times"
        raise RuntimeError(msg)

    def prompt(self, prompt: str) -> tuple[str, PromptingState]:
        prompting_state = self._get_prompting_state(prompt)

        response = self._call_generate_content_api(prompting_state)

        if len(response.candidates) == 0:
            msg = f"No message returned by model {self.name} for prompt {prompt}"
            raise RuntimeError(msg)
        candidate = response.candidates[0]

        if candidate.finish_reason == FinishReason.MAX_TOKENS:
            logger.warning("Exceeded max tokens %d", self.max_output_tokens)
        elif candidate.finish_reason != FinishReason.STOP:
            logger.warning("Finish reason %s encountered",  candidate.finish_reason.name)
            return candidate.finish_reason.name, prompting_state

        if len(candidate.content.parts) > 1:
            msg = f"Response has {len(candidate.content.parts)} parts"
            raise RuntimeError(msg)
        if len(candidate.content.parts) == 0:
            msg = "Response has no parts"
            raise RuntimeError(msg)
        output = candidate.content.parts[0].text
        return output, prompting_state

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> tuple[str, dict[str, float], PromptingState]:
        raise NotImplementedError
