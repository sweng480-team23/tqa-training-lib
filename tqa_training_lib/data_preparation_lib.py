import json
import os
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from fuzzywuzzy import fuzz, process
from normalise import normalise


def lower_case_filter(datum: dict):
    answer = datum["Answer"].lower()
    tweet = datum["Tweet"].lower()
    question = datum["Question"].lower()

    return {"Answer": answer,
            "qid": datum["qid"],
            "Question": question,
            "Tweet": tweet}


def remove_groupings_str(input: str):
    for chr in ['(', ')', '[', ']', '{', '}']:
        input = input.replace(chr, '')

    return input


def remove_groupings(datum: dict):
    answer = remove_groupings_str(datum["Answer"])
    tweet = remove_groupings_str(datum["Tweet"])
    question = remove_groupings_str(datum["Question"])

    return {
        "Answer": answer,
        "qid": datum["qid"],
        "Question": question,
        "Tweet": tweet
    }


def normalise_datum(datum: dict):
    answer = ' '.join(normalise(datum["Answer"], verbose=False))
    tweet = ' '.join(normalise(datum["Tweet"], verbose=False))
    question = ' '.join(normalise(datum["Question"], verbose=False))

    return {"Answer": answer,
            "qid": datum["qid"],
            "Question": question,
            "Tweet": tweet}


def fuzzy_match(tweet: str, answer: str) -> Tuple[int, int]:
    canidates = []
    tweet_split = tweet.split()
    answer_split = answer.split()

    n = len(answer_split)
    m = len(tweet_split)

    for i in range(m - n):
        canidates.append(tweet_split[i:i + n])

    canidates = [' '.join(canidate) for canidate in canidates]

    best_matches = process.extractBests(answer,
                                        canidates,
                                        scorer=fuzz.token_sort_ratio,
                                        score_cutoff=75)

    if best_matches:
        best_matches = [(match[1], match[0]) for match in best_matches]
        best_match = max(best_matches)[1]

        start_position = tweet.find(best_match)
        end_position = start_position + len(best_match)

        return start_position, end_position
    else:
        return -1, -1


def identify_start_and_end_positions(datum: dict) -> dict:
    tweet = datum["Tweet"]
    question = datum["Question"]
    answer = datum["Answer"]

    start_position = tweet.find(answer)

    if start_position > -1:
        end_position = start_position + len(answer)
    else:
        start_position, end_position = fuzzy_match(tweet, answer)

    assert start_position <= end_position, f'{start_position} > {end_position}'

    return {
        "qid": datum["qid"],
        "tweet": tweet,
        "question": question,
        "answer": answer,
        "start_position": start_position,
        "end_position": end_position,
    }


def add_token_positions(tokenizer, encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):

        start_positions.append(encodings.char_to_token(i, answers[i]['start_position']))
        end_positions.append(encodings.char_to_token(i, answers[i]['end_position'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def do_filters(datum: dict):
    datum = lower_case_filter(datum)
    datum = remove_groupings(datum)
    try:
        datum = normalise_datum(datum)
    except BaseException:
        print('skipped normalising due to error, qid = ' + datum['qid'])
        pass
    datum = identify_start_and_end_positions(datum)
    return datum


def prepare_data(df: pd.DataFrame, save_data=False, print_stats=False):
    df["Answer"] = df["Answer"].explode()
    train_data, val_data = train_test_split(df, test_size=0.2)
    x_train_before = train_data.to_dict('records')
    x_val_before = val_data.to_dict('records')

    x_train_filtered = [do_filters(datum) for datum in x_train_before]
    x_val_filtered = [do_filters(datum) for datum in x_val_before]

    non_quality_x_train = [datum for datum in x_train_filtered if datum["start_position"] == -1]
    non_quality_x_val = [datum for datum in x_val_filtered if datum["start_position"] == -1]
    quality_x_train = [datum for datum in x_train_filtered if datum["start_position"] >= 0]
    quality_x_val = [datum for datum in x_val_filtered if datum["start_position"] >= 0]

    if print_stats:
        stat_dict = {
            "Quality Train": len(quality_x_train),
            "Quality Val": len(quality_x_val),
            "Bad Train": len(non_quality_x_train),
            "Bad Val": len(non_quality_x_val),
        }
        ratio = (len(quality_x_train) + len(quality_x_val)) / (len(non_quality_x_train) + len(non_quality_x_val))
        stat_df = pd.DataFrame(stat_dict.items(), columns=['Stat', 'Value'])

        print('Quality ratio (higher is better): %.4f' % ratio)
        print(stat_df.to_markdown())

    if save_data:
        if not os.path.exists('prep'):
            os.mkdir('prep')
        with open('prep/x_train_before.json', 'w') as f_out:
            json.dump(x_train_before, f_out)
        with open('prep/x_train_filtered.json', 'w') as f_out:
            json.dump(x_train_filtered, f_out)
        with open('prep/quality_x_train.json', 'w') as f_out:
            json.dump(quality_x_train, f_out)

    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    train_encodings = tokenizer(
        [q["tweet"] for q in quality_x_train],
        [q["question"] for q in quality_x_train],
        max_length=50,
        padding='max_length',
        truncation=True)

    val_encodings = tokenizer(
        [q["tweet"] for q in quality_x_val],
        [q["question"] for q in quality_x_val],
        max_length=50,
        padding='max_length',
        truncation=True)

    add_token_positions(tokenizer, train_encodings, quality_x_train)
    add_token_positions(tokenizer, val_encodings, quality_x_val)

    return train_encodings, val_encodings
