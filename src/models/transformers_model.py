import logging
import os

from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from models.model import Model, ModelName

logger = logging.getLogger(__name__)


class TransformersModel(Model):
    SUPPORTED_MODELS = (
        ModelName.AYA_101,
        ModelName.GEMMA_7B,
        ModelName.MIXTRAL_8X7B,
        ModelName.XGLM_7_5B,
    )

    def __init__(self,
                 name: ModelName,
                 gpus: list[int] | None = None,
                 max_mem_per_gpu: int | None = None,
                 **kwargs) -> None:
        super().__init__(name)
        if name not in TransformersModel.SUPPORTED_MODELS:
            msg = f"{name} is not a valid transformers model"
            raise ValueError(msg)

        load_dotenv()

        self.tokenizer = TransformersModel._get_tokenizer(self.name)
        self.model = TransformersModel._get_model(self.name, gpus, max_mem_per_gpu)

    @staticmethod
    def _get_pretrained_model_name_or_path(model_name: ModelName) -> str:
        if model_name == ModelName.GEMMA_7B:
            return "google/gemma-7b"
        if model_name == ModelName.XGLM_7_5B:
            return "facebook/xglm-7.5B"
        if model_name == ModelName.MIXTRAL_8X7B:
            return "mistralai/Mixtral-8x7B-Instruct-v0.1"
        if model_name == ModelName.AYA_101:
            return "CohereForAI/aya-101"
        raise NotImplementedError

    @staticmethod
    def _get_tokenizer(model_name: ModelName) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        model_name_or_path = TransformersModel._get_pretrained_model_name_or_path(model_name)
        token = os.environ.get("HF_USER_ACCESS_TOKEN")

        return AutoTokenizer.from_pretrained(model_name_or_path, token=token)

    @staticmethod
    def _get_model(model_name: ModelName, gpus: list[int] | None, max_mem_per_gpu: int | None = None) -> PreTrainedModel:
        gpus = gpus or []

        model_name_or_path = TransformersModel._get_pretrained_model_name_or_path(model_name)

        max_memory = {i: 0 for i in range(torch.cuda.device_count())}
        for gpu in gpus:
            max_memory[gpu] = max_mem_per_gpu if max_mem_per_gpu is not None else torch.cuda.get_device_properties(gpu).total_memory
        device_map = "auto" if len(gpus) > 0 else None
        token = os.environ.get("HF_USER_ACCESS_TOKEN")

        if model_name == ModelName.AYA_101:
            return AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path, device_map=device_map, max_memory=max_memory, token=token)

        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=device_map, max_memory=max_memory, token=token)

    def prompt(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**input_ids)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
