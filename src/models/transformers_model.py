from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Self

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic.dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.generation.utils import GenerateOutput

from models.model import Model, ModelName, PromptingState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformersPromptParameters(PromptingState):
    model_path: str | None = None
    model_input_dict: dict | None = None
    output_scores: bool = False
    return_dict_in_generate: bool = False


class TransformersModel(Model):
    SUPPORTED_MODELS = (
        ModelName.AYA_101,
        ModelName.GEMMA_7B,
        ModelName.MIXTRAL_8X7B,
        ModelName.XGLM_7_5B,
    )

    def __init__(
        self,
        name: ModelName,
        max_output_tokens: int,
        **_,
    ) -> None:
        super().__init__(name, max_output_tokens)
        if name not in TransformersModel.SUPPORTED_MODELS:
            msg = f"{name} is not a valid transformers model"
            raise ValueError(msg)

        load_dotenv()

        self.token = os.environ.get("HF_USER_ACCESS_TOKEN")

    @classmethod
    def make(
        cls: type[Self],
        model_name: ModelName,
        max_output_tokens: int,
        **kwargs,
    ) -> TransformersModel:
        if model_name == ModelName.AYA_101:
            return Aya101Model(model_name, max_output_tokens, **kwargs)
        if model_name == ModelName.GEMMA_7B:
            return Gemma7BModel(model_name, max_output_tokens, **kwargs)
        if model_name == ModelName.MIXTRAL_8X7B:
            return Mixtral8x7BModel(model_name, max_output_tokens, **kwargs)
        if model_name == ModelName.XGLM_7_5B:
            return Xglm7Pt5BModel(model_name, max_output_tokens, **kwargs)
        raise NotImplementedError(model_name)

    def _init_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        default_parameters = self.default_parameters
        assert isinstance(default_parameters, TransformersPromptParameters)
        assert default_parameters.model_path is not None

        return AutoTokenizer.from_pretrained(default_parameters.model_path, token=self.token)

    def _init_model(
        self,
        gpus: list[int] | None = None,
        max_mem_per_gpu: int | None = None,
    ) -> PreTrainedModel:
        max_memory = self._get_max_memory_map(gpus, max_mem_per_gpu)
        device_map = "auto" if gpus is not None and len(gpus) > 0 else None

        default_parameters = self.default_parameters
        assert isinstance(default_parameters, TransformersPromptParameters)
        assert default_parameters.model_path is not None

        return AutoModelForCausalLM.from_pretrained(
            default_parameters.model_path,
            device_map=device_map,
            max_memory=max_memory,
            token=self.token,
        )

    def _get_prompting_state(
        self,
        prompt: str,
        max_output_tokens: int | None = None,
        **prompting_state_kwargs,
    ) -> TransformersPromptParameters:
        prompt_params_dict = dataclasses.asdict(self.default_parameters)
        prompt_params_dict["prompt"] = prompt
        prompt_params_dict["max_output_tokens"] = (
            max_output_tokens if max_output_tokens is not None else self.max_output_tokens
        )

        return TransformersPromptParameters(**prompt_params_dict, **prompting_state_kwargs)

    def _get_max_memory_map(
        self,
        gpus: list[int] | None,
        max_mem_per_gpu: int | None = None,
    ) -> dict[int, int]:
        gpus = gpus or []

        max_memory = {i: 0 for i in range(torch.cuda.device_count())}
        for gpu in gpus:
            max_memory[gpu] = (
                max_mem_per_gpu
                if max_mem_per_gpu is not None
                else torch.cuda.get_device_properties(gpu).total_memory
            )

        return max_memory

    def _call_generate(
        self,
        model: PreTrainedModel,
        prompting_state: TransformersPromptParameters,
    ) -> GenerateOutput | torch.LongTensor:
        if prompting_state.model_input_dict is None:
            msg = "Model input dictionary cannot be None"
            raise ValueError(msg)

        return model.generate(
            **prompting_state.model_input_dict,
            max_new_tokens=prompting_state.max_output_tokens,
        )


class Gemma7BModel(TransformersModel):
    DEFAULT_PARAMETERS = TransformersPromptParameters(
        prompt=None,
        model_name=ModelName.GEMMA_7B,
        max_output_tokens=2048,
        model_path="google/gemma-1.1-7b-it",
    )

    def __init__(
        self,
        name: ModelName,
        max_output_tokens: int,
        gpus: list[int] | None = None,
        max_mem_per_gpu: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, max_output_tokens, **kwargs)

        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model(gpus, max_mem_per_gpu)
        self._default_parameters = TransformersPromptParameters(
            prompt=None,
            model_name=name,
            max_output_tokens=max_output_tokens,
            model_path="google/gemma-1.1-7b-it",
        )

    @property
    def default_parameters(self) -> PromptingState:
        return self._default_parameters

    def _get_prompting_state(
        self,
        prompt: str,
        batch_encoding: BatchEncoding | None = None,
        **prompting_state_kwargs,
    ) -> TransformersPromptParameters:
        if batch_encoding is None:
            batch_encoding = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device,
            )

        return super()._get_prompting_state(
            prompt,
            model_input_dict=batch_encoding,
            **prompting_state_kwargs,
        )

    def get_prompting_state(self, prompt: str) -> PromptingState:
        return self._get_prompting_state(prompt)

    def prompt(self, prompt: str) -> tuple[str, PromptingState]:
        batch_encoding: BatchEncoding = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.device,
        )

        prompting_state = self._get_prompting_state(
            prompt,
            batch_encoding=batch_encoding,
        )

        outputs = self._call_generate(self.model, prompting_state)

        return self.tokenizer.decode(
            outputs[0, batch_encoding.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ), prompting_state

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> tuple[str, dict[str, float], PromptingState]:
        batch_encoding = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        prompting_state = self._get_prompting_state(
            prompt,
            batch_encoding=batch_encoding,
            max_output_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )

        outputs = self._call_generate(self.model, prompting_state)
        assert isinstance(outputs, GenerateOutput)
        assert outputs.scores is not None

        generated_tokens = outputs.sequences[:, batch_encoding.input_ids.shape[1] :]
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )

        token_probabilities = {}
        for token_id, score in zip(generated_tokens[0], transition_scores[0]):
            token_str = self.tokenizer.decode(
                token_id,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            token_probabilities[token_str] = np.exp(score)

        output_str = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        return output_str, token_probabilities, prompting_state


class Mixtral8x7BModel(TransformersModel):
    DEFAULT_PARAMETERS = TransformersPromptParameters(
        prompt=None,
        model_name=ModelName.MIXTRAL_8X7B,
        max_output_tokens=2048,
        model_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )

    def __init__(
        self,
        name: ModelName,
        max_output_tokens: int,
        gpus: list[int] | None = None,
        max_mem_per_gpu: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, max_output_tokens, **kwargs)

        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model(gpus, max_mem_per_gpu)
        self._default_parameters = TransformersPromptParameters(
            prompt=None,
            model_name=name,
            max_output_tokens=max_output_tokens,
            model_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )

    @property
    def default_parameters(self) -> PromptingState:
        return self._default_parameters

    def _get_prompting_state(
        self,
        prompt: str,
        inputs: Any | None = None,
        **prompting_state_kwargs,
    ) -> TransformersPromptParameters:
        if inputs is None:
            messages = [
                {"role": "user", "content": prompt},
            ]

            inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

        return super()._get_prompting_state(
            prompt,
            model_input_dict={"inputs": inputs},
            **prompting_state_kwargs,
        )

    def get_prompting_state(self, prompt: str) -> PromptingState:
        return self._get_prompting_state(prompt)

    def prompt(self, prompt: str) -> tuple[str, PromptingState]:
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

        prompting_state = self._get_prompting_state(prompt, inputs=inputs)

        outputs = self._call_generate(self.model, prompting_state)
        return self.tokenizer.decode(
            outputs[0, inputs.shape[1] :],
            skip_special_tokens=True,
        ), prompting_state

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> tuple[str, dict[str, float], PromptingState]:
        messages = [
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

        prompting_state = self._get_prompting_state(
            prompt,
            inputs=inputs,
            max_output_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )

        outputs = self._call_generate(self.model, prompting_state)
        assert isinstance(outputs, GenerateOutput)
        assert outputs.scores is not None

        generated_tokens = outputs.sequences[:, inputs.shape[1] :]
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )

        token_probabilities = {}
        for token_id, score in zip(generated_tokens[0], transition_scores[0]):
            token_str = self.tokenizer.decode(
                token_id,
                skip_special_tokens=True,
            )
            token_probabilities[token_str] = np.exp(score)

        output_str = self.tokenizer.decode(
            outputs[0, inputs.shape[1] :],
            skip_special_tokens=True,
        )
        return output_str, token_probabilities, prompting_state


class Xglm7Pt5BModel(TransformersModel):
    DEFAULT_PARAMETERS = TransformersPromptParameters(
        prompt=None,
        model_name=ModelName.XGLM_7_5B,
        max_output_tokens=2048,
        model_path="facebook/xglm-7.5B",
    )

    def __init__(
        self,
        name: ModelName,
        max_output_tokens: int,
        gpus: list[int] | None = None,
        max_mem_per_gpu: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, max_output_tokens, **kwargs)

        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model(gpus, max_mem_per_gpu)
        self._default_parameters = TransformersPromptParameters(
            prompt=None,
            model_name=name,
            max_output_tokens=max_output_tokens,
            model_path="facebook/xglm-7.5B",
        )

    @property
    def default_parameters(self) -> PromptingState:
        return self._default_parameters

    def get_prompting_state(self, prompt: str) -> PromptingState:
        raise NotImplementedError

    def prompt(self, prompt: str) -> tuple[str, PromptingState]:
        raise NotImplementedError

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> tuple[str, dict[str, float], PromptingState]:
        raise NotImplementedError


class Aya101Model(TransformersModel):
    DEFAULT_PARAMETERS = TransformersPromptParameters(
        prompt=None,
        model_name=ModelName.AYA_101,
        max_output_tokens=2048,
        model_path="CohereForAI/aya-101",
    )

    def __init__(
        self,
        name: ModelName,
        max_output_tokens: int,
        gpus: list[int] | None = None,
        max_mem_per_gpu: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, max_output_tokens, **kwargs)

        self._default_parameters = TransformersPromptParameters(
            prompt=None,
            model_name=name,
            max_output_tokens=max_output_tokens,
            model_path="CohereForAI/aya-101",
        )
        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model(gpus, max_mem_per_gpu)

    @property
    def default_parameters(self) -> PromptingState:
        return self._default_parameters

    def _init_model(
        self,
        gpus: list[int] | None = None,
        max_mem_per_gpu: int | None = None,
    ) -> PreTrainedModel:
        max_memory = self._get_max_memory_map(gpus, max_mem_per_gpu)
        device_map = "auto" if gpus is not None and len(gpus) > 0 else None

        default_parameters = self.default_parameters
        assert isinstance(default_parameters, TransformersPromptParameters)
        assert default_parameters.model_path is not None

        return AutoModelForSeq2SeqLM.from_pretrained(
            default_parameters.model_path,
            device_map=device_map,
            max_memory=max_memory,
            token=self.token,
        )

    def _get_prompting_state(
        self,
        prompt: str,
        batch_encoding: BatchEncoding | None = None,
        **prompting_state_kwargs,
    ) -> TransformersPromptParameters:
        if batch_encoding is None:
            batch_encoding = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device,
            )

        prompting_state_kwargs["model_input_dict"] = batch_encoding

        return super()._get_prompting_state(
            prompt,
            **prompting_state_kwargs,
        )

    def get_prompting_state(self, prompt: str) -> PromptingState:
        return self._get_prompting_state(prompt)

    def prompt(self, prompt: str) -> tuple[str, PromptingState]:
        batch_encoding: BatchEncoding = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.device,
        )

        prompting_state = self._get_prompting_state(
            prompt,
            batch_encoding=batch_encoding,
        )

        outputs = self._call_generate(self.model, prompting_state)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True), prompting_state

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> tuple[str, dict[str, float], PromptingState]:
        batch_encoding = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        prompting_state = self._get_prompting_state(
            prompt,
            batch_encoding=batch_encoding,
            max_output_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )

        outputs = self._call_generate(self.model, prompting_state)
        assert isinstance(outputs, GenerateOutput)
        assert outputs.scores is not None

        generated_tokens = outputs.sequences
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )

        token_probabilities = {}
        for token_id, score in zip(generated_tokens[0], transition_scores[0]):
            token_str = self.tokenizer.decode(
                token_id,
                skip_special_tokens=True,
            )
            token_probabilities[token_str] = np.exp(score)

        output_str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return output_str, token_probabilities, prompting_state
