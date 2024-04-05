import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from models.model import Model, ModelName

logger = logging.getLogger(__name__)


class TransformersModel(Model):
    SUPPORTED_MODELS = (
        ModelName.AYA_101,
        ModelName.GEMMA_7B,
        ModelName.MIXTRAL_8X7B,
        ModelName.XGLM_7_5B,
    )

    def __init__(self, name: ModelName) -> None:
        super().__init__(name)
        if name not in TransformersModel.SUPPORTED_MODELS:
            msg = f"{name} is not a valid transformers model"
            raise ValueError(msg)

        self.pretrained_model_name_or_path = self._get_pretrained_model_name_or_path()
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name_or_path,
            device_map="auto",
        )

    def _get_pretrained_model_name_or_path(self) -> str:
        if self.name == ModelName.GEMMA_7B:
            return "google/gemma-7b"
        if self.name == ModelName.XGLM_7_5B:
            return "facebook/xglm-7.5B"
        if self.name == ModelName.MIXTRAL_8X7B:
            return "mistralai/Mixtral-8x7B-Instruct-v0.1"
        if self.name == ModelName.AYA_101:
            return "CohereForAI/aya-101"
        raise NotImplementedError

    def prompt(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**input_ids)
        return self.tokenizer.decode(outputs)
