from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class ModelRunner(ABC):
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel

    def __init__(self, model_path: str, pretrained_base: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_base)
        self.model = AutoModel.from_pretrained(model_path)
        super().__init__()

    @abstractmethod
    def answer_tweet_question(self, tweet, question) -> tuple[str, int, int]:
        pass
