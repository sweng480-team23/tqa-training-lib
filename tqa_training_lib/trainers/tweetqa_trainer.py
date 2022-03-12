from abc import ABC, abstractmethod

from tqa_training_lib.trainers.tweetqa_training_args import TweetQATrainingArgs


class TweetQATrainer(ABC):
    @abstractmethod
    def train(self, train_encodings, val_encodings, args: TweetQATrainingArgs) -> any:
        pass
