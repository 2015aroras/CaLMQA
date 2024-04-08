import logging

from dotenv import load_dotenv
from openai import OpenAI

from models.model import Model, ModelName

logger = logging.getLogger(__name__)


class OpenAIModel(Model):
    MAX_TOKENS = 2048
    SUPPORTED_MODELS = (ModelName.GPT_4,)

    def __init__(self, name: ModelName, **kwargs) -> None:
        super().__init__(name)
        if name not in OpenAIModel.SUPPORTED_MODELS:
            msg = f"{name} is not a valid OpenAI model"
            raise ValueError(msg)

        load_dotenv()
        self.client = OpenAI()

    def _get_model_param(self) -> str:
        if self.name == ModelName.GPT_4:
            return "gpt-4-0125-preview"
        raise NotImplementedError

    def prompt(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            max_tokens=OpenAIModel.MAX_TOKENS,
            model=self._get_model_param(),
            messages=[{"role": "user", "content": prompt}],
        )

        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning("Exceeded max tokens %d", OpenAIModel.MAX_TOKENS)
        elif choice.finish_reason != "stop":
            msg = f"Unexpected finish reason {choice.finish_reason}"
            raise RuntimeError(msg)

        if choice.message.content is None:
            msg = f"No message returned by model {self.name} for prompt {prompt}"
            raise RuntimeError(msg)

        return choice.message.content
