import re
import tensorflow as tf

from tqa_training_lib.model_runners.model_runner import ModelRunner


class TFBertModelRunner(ModelRunner):

    def __init__(self, model_path: str, pretrained_base: str) -> None:
        super().__init__(model_path, pretrained_base)

    def answer_tweet_question(self, tweet, question) -> tuple[str, int, int]:
        input_dict = self.tokenizer(question, tweet, return_tensors="tf")
        outputs = self.model(input_dict)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        answer = " ".join(all_tokens[tf.math.argmax(start_logits, 1)[0]:tf.math.argmax(end_logits, 1)[0] + 1])
        answer_fixed = re.sub(r'\s##', '', answer)
        return answer_fixed, tf.math.argmax(start_logits, 1)[0], tf.math.argmax(end_logits, 1)[0] + 1
