import torch
from transformers import AutoTokenizer, BertForQuestionAnswering
from tqa_training_lib.model_runners.model_runner import ModelRunner


class TorchBertModelRunner(ModelRunner):

    def __init__(self, model_path: str, pretrained_base: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_base)
        self.model = BertForQuestionAnswering.from_pretrained(model_path)
        super().__init__()

    def answer_tweet_question(self, tweet, question) -> tuple[str, int, int]:
        tweet = tweet.lower()
        question = question.lower()
        # tokenize question and text as a pair
        input_ids = self.tokenizer.encode(question, tweet)

        # string version of tokenized ids
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # segment IDs
        # first occurence of [SEP] token
        sep_idx = input_ids.index(self.tokenizer.sep_token_id)

        # number of tokens in segment A (question)
        num_seg_a = sep_idx + 1
        # number of tokens in segment B (text)
        num_seg_b = len(input_ids) - num_seg_a

        # list of 0s and 1s for segment embeddings
        segment_ids = [1] * num_seg_a + [0] * num_seg_b
        assert len(segment_ids) == len(input_ids)

        # model output using input_ids and segment_ids
        self.model.eval()
        output = self.model(input_ids=torch.tensor([input_ids]), attention_mask=torch.tensor([segment_ids]))

        # reconstructing the answer
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits)

        if answer_end >= answer_start:
            answer = tokens[answer_start]
            for i in range(answer_start + 1, answer_end + 1):
                if tokens[i][0:2] == "##":
                    answer += tokens[i][2:]
                else:
                    answer += " " + tokens[i]
        else:
            answer = "Unable to find the answer to your question."

        return answer, answer_start.item(), answer_end.item()
