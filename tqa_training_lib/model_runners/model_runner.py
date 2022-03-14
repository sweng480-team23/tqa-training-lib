from abc import ABC, abstractmethod
from typing import Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer


class ModelRunner(ABC):
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def answer_tweet_question(self, tweet, question) -> Tuple[str, int, int]:
        pass
