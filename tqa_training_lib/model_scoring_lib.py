import json
import os
import shutil
import pandas as pd
import numpy as np
import string
import re
import torch
import tensorflow as tf

from typing import List
from transformers import BertModel, BertTokenizer, BertTokenizerFast, TFAutoModelForQuestionAnswering, TFBertForQuestionAnswering, TFBertModel
from nltk.translate.bleu_score import sentence_bleu
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class BertModelRunner(object):
    tokenizer: BertTokenizer
    model: BertModel

    def __init__(self, tokenizer: BertTokenizer, model: BertModel) -> None:
        self.tokenizer = tokenizer
        self.model = model
        super().__init__()

    def answer_tweet_question(self, tweet, question):
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


def read_data(url: str) -> pd.DataFrame:
    return pd.read_json(url)


def get_model_runner(model_path: str) -> BertModelRunner:
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    return BertModelRunner(tokenizer, model)


def generate_gold_file(df: pd.DataFrame) -> List[dict]:
    data_dict: dict = df.to_dict('records')
    return [{'qid': datum['qid'], 'Answer': datum['Answer']} for datum in data_dict]


def to_prediction(datum: dict, model_runner: BertModelRunner) -> dict:
    answer, start, end = model_runner.answer_tweet_question(datum['Tweet'], datum['Question'])
    return {
        'qid': datum['qid'],
        'Tweet': datum['Tweet'],
        'Question': datum['Question'],
        'Answer': answer,
        'Start': start,
        'End': end,
        'Actual Answer': datum['Answer']
    }


def generate_user_file(df: pd.DataFrame, model_path: str) -> List[dict]:
    model_runner: BertModelRunner = get_model_runner(model_path)
    data_dict: dict = df.to_dict('records')
    return [to_prediction(datum, model_runner) for datum in data_dict]


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def ans_score(ans, gold_list, meteor_scorer, rouge_scorer):
    ans = normalize_answer(ans)
    gold_list = [normalize_answer(ref) for ref in gold_list]
    bleu = sentence_bleu([_.split() for _ in gold_list], ans.split(), weights=(1, 0, 0, 0))
    meteor, _ = meteor_scorer.compute_score({0: gold_list}, {0: [ans]})
    rouge, _ = rouge_scorer.compute_score({0: gold_list}, {0: [ans]})
    return {'bleu': bleu, 'meteor': meteor, 'rouge': rouge}


def evaluate(gold, pred, meteor_scorer, rouge_scorer):
    idx2gold = {item['qid']: item['Answer'] for item in gold}
    idx2pred = {item['qid']: item['Answer'] for item in pred}
    idx2scores = {}
    for id_ in idx2gold.keys():
        if isinstance(idx2pred[id_], list):
            pred_ans = idx2pred[id_][0]
        else:
            pred_ans = idx2pred[id_]
        idx2scores[id_] = ans_score(pred_ans, idx2gold[id_], meteor_scorer, rouge_scorer)
    bleus = [item['bleu'] for item in idx2scores.values()]
    meteors = [item['meteor'] for item in idx2scores.values()]
    rouges = [item['rouge'] for item in idx2scores.values()]

    return {
        'BLEU-1': np.mean(bleus),
        'METEOR': np.mean(meteors),
        'ROUGE': np.mean(rouges)
    }


def generate_user_file_tf(df: pd.DataFrame, model_path: str) -> List[dict]:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    model = TFBertForQuestionAnswering.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    data_dict: dict = df.to_dict('records')
    return [to_prediction_tf(datum, model, tokenizer) for datum in data_dict]


def to_prediction_tf(datum: dict, model: TFBertModel, tokenizer: BertTokenizer) -> dict:
    answer = answer_tweet_question_tf(model, tokenizer, datum['Tweet'], datum['Question'])

    return {
        'qid': datum['qid'],
        'Tweet': datum['Tweet'],
        'Question': datum['Question'],
        'Answer': answer,
        'Actual Answer': datum['Answer']
    }


def answer_tweet_question_tf(model: TFBertForQuestionAnswering, tokenizer: BertTokenizer, tweet, question):
    input_dict = tokenizer(question, tweet, return_tensors="tf")
    outputs = model(input_dict)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
    answer = " ".join(all_tokens[tf.math.argmax(start_logits, 1)[0]:tf.math.argmax(end_logits, 1)[0] + 1])
    return answer


def score_model(model_path: str, save_gold_user_files=False, print_scores=False, use_tf=False):
    data: pd.DataFrame = read_data('https://raw.githubusercontent.com/sweng480-team23/tweet-qa-data/main/dev.json')
    gold_file = generate_gold_file(data)
    if use_tf:
        user_file = generate_user_file_tf(data, model_path)
    else:
        user_file = generate_user_file(data, model_path)

    meteor_scorer = Meteor()
    rouge_scorer = Rouge()
    scores = evaluate(gold_file, user_file, meteor_scorer, rouge_scorer)

    if print_scores:
        print(scores)

    if save_gold_user_files:
        if os.path.exists('scoring/'):
            shutil.rmtree('scoring/')

        os.mkdir('scoring/')
        with open('scoring/gold_file.json', 'w') as f_out:
            json.dump(gold_file, f_out)
        with open('scoring/user_file.json', 'w') as f_out:
            json.dump(user_file, f_out)

    return scores
