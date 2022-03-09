import json
import os
import shutil
import pandas as pd
import numpy as np
import string
import re
import tensorflow as tf

from typing import List
from nltk.translate.bleu_score import sentence_bleu
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge

from tqa_training_lib.model_runners.model_runner import ModelRunner
from tqa_training_lib.model_runners.tf_bert_model_runner import TFBertModelRunner
from tqa_training_lib.model_runners.torch_bert_model_runner import TorchBertModelRunner


def read_data(url: str) -> pd.DataFrame:
    return pd.read_json(url)


def generate_gold_file(df: pd.DataFrame) -> List[dict]:
    data_dict: dict = df.to_dict('records')
    return [{'qid': datum['qid'], 'Answer': datum['Answer']} for datum in data_dict]


def to_prediction(datum: dict, runner: ModelRunner) -> dict:
    answer, start, end = runner.answer_tweet_question(datum['Tweet'], datum['Question'])
    return {
        'qid': datum['qid'],
        'Tweet': datum['Tweet'],
        'Question': datum['Question'],
        'Answer': answer,
        'Start': start,
        'End': end,
        'Actual Answer': datum['Answer']
    }


def generate_user_file(df: pd.DataFrame, runner: ModelRunner) -> List[dict]:
    data_dict: dict = df.to_dict('records')
    return [to_prediction(datum, runner) for datum in data_dict]


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


def score_model(model_path: str, save_gold_user_files=False, print_scores=False, use_tf=False):
    data: pd.DataFrame = read_data('https://raw.githubusercontent.com/sweng480-team23/tweet-qa-data/main/dev.json')
    gold_file = generate_gold_file(data)

    if use_tf:
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

        runner = TFBertModelRunner(model_path, 'bert-large-uncased-whole-word-masking-finetuned-squad')
    else:
        runner = TorchBertModelRunner(model_path, 'bert-large-uncased-whole-word-masking-finetuned-squad')

    user_file = generate_user_file(data, runner)

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
