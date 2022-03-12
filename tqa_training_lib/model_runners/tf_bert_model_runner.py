import re
from numpy import dtype
import tensorflow as tf
from transformers import AutoTokenizer, TFBertForQuestionAnswering

from tqa_training_lib.model_runners.model_runner import ModelRunner


class TFBertModelRunner(ModelRunner):

    def __init__(self, model_path: str, pretrained_base: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_base)
        self.model = TFBertForQuestionAnswering.from_pretrained(model_path)
        super().__init__()

    def answer_tweet_question(self, tweet, question) -> tuple[str, int, int]:
        input_dict = self.tokenizer(question, tweet, return_tensors="tf")
        outputs = self.model(input_dict)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        answer = " ".join(all_tokens[tf.math.argmax(start_logits, 1)[0]:tf.math.argmax(end_logits, 1)[0] + 1])
        answer_fixed = re.sub(r'\s##', '', answer)
        print(answer_fixed)
        # todo: return actual start and end
        return answer_fixed, 0, 0
