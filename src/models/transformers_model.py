from __future__ import annotations

import logging
import os
from abc import abstractmethod
from typing import Self

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.generation.utils import GenerateOutput

from models.model import Model, ModelName

logger = logging.getLogger(__name__)


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
        **kwargs,
    ) -> None:
        super().__init__(name, max_output_tokens)
        if name not in TransformersModel.SUPPORTED_MODELS:
            msg = f"{name} is not a valid transformers model"
            raise ValueError(msg)

        load_dotenv()

        self.token = os.environ.get("HF_USER_ACCESS_TOKEN")

    @property
    @abstractmethod
    def model_path(self) -> str:
        pass

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
        return AutoTokenizer.from_pretrained(self.model_path, token=self.token)

    def _init_model(
        self,
        gpus: list[int] | None = None,
        max_mem_per_gpu: int | None = None,
    ) -> PreTrainedModel:
        max_memory = self._get_max_memory_map(gpus, max_mem_per_gpu)
        device_map = "auto" if gpus is not None and len(gpus) > 0 else None

        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            max_memory=max_memory,
            token=self.token,
        )

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


class Gemma7BModel(TransformersModel):
    MODEL_PATH = "google/gemma-7b-it"

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

    @property
    def model_path(self) -> str:
        return self.MODEL_PATH

    def prompt(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**input_ids, max_new_tokens=self.max_output_tokens)
        return self.tokenizer.decode(
            outputs[0, input_ids["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 5,
    ) -> tuple[str, dict[str, float]]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        assert isinstance(outputs, GenerateOutput)
        assert outputs.scores is not None

        generated_tokens = outputs.sequences[:, input_ids.input_ids.shape[1] :]
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
            outputs[0, input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return output_str, token_probabilities


class Mixtral8x7BModel(TransformersModel):
    MODEL_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"

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

    @property
    def model_path(self) -> str:
        return self.MODEL_PATH

    def prompt(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

        outputs = self.model.generate(inputs, max_new_tokens=self.max_output_tokens)
        return self.tokenizer.decode(
            outputs[0, inputs.shape[1] :],
            skip_special_tokens=True,
        )

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 5,
    ) -> tuple[str, dict[str, float]]:
        messages = [
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
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
        return output_str, token_probabilities


class Xglm7Pt5BModel(TransformersModel):
    MODEL_PATH = "facebook/xglm-7.5B"

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

    @property
    def model_path(self) -> str:
        return self.MODEL_PATH

    def prompt(self, prompt: str) -> str:
        raise NotImplementedError

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 5,
    ) -> tuple[str, dict[str, float]]:
        raise NotImplementedError


class Aya101Model(TransformersModel):
    MODEL_PATH = "CohereForAI/aya-101"

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

    @property
    def model_path(self) -> str:
        return self.MODEL_PATH

    def _init_model(
        self,
        gpus: list[int] | None = None,
        max_mem_per_gpu: int | None = None,
    ) -> PreTrainedModel:
        max_memory = self._get_max_memory_map(gpus, max_mem_per_gpu)
        device_map = "auto" if gpus is not None and len(gpus) > 0 else None

        return AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            max_memory=max_memory,
            token=self.token,
        )

    def prompt(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**input_ids, max_new_tokens=self.max_output_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def prompt_and_next_token_probs(
        self,
        prompt: str,
        max_new_tokens: int = 5,
    ) -> tuple[str, dict[str, float]]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
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

        return output_str, token_probabilities
