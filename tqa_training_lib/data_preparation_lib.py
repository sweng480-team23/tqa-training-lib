import json
import os
import re
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from fuzzywuzzy import fuzz, process
from normalise import normalise

from tqa_training_lib.model_scoring_lib import normalize_answer

qid_blacklist = [
    'e1b0e8a6ba8d16b51c260e04f5a65561',
    'd6cd5a331eed025bb8c39e13a59b7267',
    '4f5c21d8ed9fee5fb89499ba6bb2e37c',
    'da59b95b1e6f2d9160ea59576e8fc3ce',
    '5dca90643c1741b3b2873d8d56dd998f',
    'bb7230322a0322416d27f01aed8ba48c',
    '8775f0531f8f8b5170f7806f3f45aca8',
]

abbrevs = {
    'tbh': 'to be honest',
    '&': 'and',
    'realdonaldtrump': 'donald trump',
    'lewishamilton': 'lewishamilton',
    'luis16suarez': 'luis suarez',
    'demconvention': 'dem convention',
    'hillaryclinton': 'hillary clinton',
    'jtimberlake': 'justin timberlake',
    'senrandpaul': 'senator rand paul',
    'randpaul': 'senator rand paul',
    'azizansari': 'aziz ansari',
    'cassinisaturn': 'cassini',
    'cnnpolitics': 'cnn',
    'vp': 'vice president',
    'piersmorgan': 'piers morgan',
    'theellenshow': 'the ellen show',
    'lenadunham': 'lena dunham',
    'nytimes': 'new york times',
    'jdsutter': 'john d sutter',
    'potus': 'president of the united states',
    'kimkardashian': 'kim kardashian',
    'justinbieber': 'justin bieber',
    'nickiminaj': 'nicki minaj',
    'jk_rowling': 'jk rowling',
    'jadapsmith': 'jada pinkett smith',
    'khloekardashian': 'khloe kardashian',
    'jebbush': 'jeb bush',
    'tedcruz': 'ted cruz',
    'whitehouse': 'white house',
    'wojyahoonba': 'yahoo nba',
    'cnnfc': 'cnn football',
    'chrissyteigen': 'chrissy teigen',
    'kensingtonroyal': 'duke and duchess of cambridge',
    'janetmock': 'janet mock',
    'thatgirlcarly': 'carly mallenbaum',
    'chrisrock': 'chris rock',
    'taylorswift13': 'taylor swift',
    'kingjames': 'lebron james',
    'chris_broussard': 'chris broussard',
    'marcorubio': 'marco rubio',
    'lesdoggg': 'leslie jones',
    'katyperry': 'katy perry',
    'senjohnmccain': 'senator john mccain',
    'mcilroyrory': 'rory mcilroy',
    'film114': 'ryan case',
    'mittromney': 'mitt romney',
    'johnlegend': 'john legend',
    'ivancnn': 'ivan watson',
    'justinkirkland4': 'justin kirkland',
    'barackobama': 'barack obama',
    'pattyarquette': 'patricia arquette',
    'goldenglobes': 'golden globes',
    'snl': 'saturday night live',
    'cnnworldcup': 'world cup',
    'worldcup': 'world cup',
    'rip': 'rest in peace',
    'irandeal': 'iran deal',
    'blacklivesmatter': 'black lives matter',
    'sochi2014': 'olympics',
    'gopconvention': 'gop convention',
    'rippaulwalker': 'paul walker',
    'thanksmichelleobama': 'thanks michelle obama',
    'greysanatomy': 'greys anatomy',
    'thesoundofmusiclive': 'sound of music',
    'facebookdown': 'facebook down',
    'nyc': 'new york city',
    'ericgarner': 'eric garner',
    'charliehebdo': 'charlie hebdo',
    'ryanreynolds': 'ryan reynolds',
    'blakelively': 'blake lively',
    'billclinton': 'bill clinton',
}


def do_abbrevs(datum: dict):
    answer = datum["Answer"]
    tweet = datum["Tweet"]
    question = datum["Question"]

    for abbrev in abbrevs:
        replacement = ' ' + abbrevs[abbrev] + ' '
        answer = answer.replace(abbrev, replacement)
        tweet = tweet.replace(abbrev, replacement)
        question = question.replace(abbrev, replacement)

    return {"Answer": answer,
            "qid": datum["qid"],
            "Question": question,
            "Tweet": tweet}


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
        input = input.replace(chr, ' ')

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


def remove_quotes(datum: dict):
    answer = datum["Answer"].replace('"', ' ')
    tweet = datum["Tweet"].replace('"', ' ')
    question = datum["Question"].replace('"', ' ')

    return {
        "Answer": answer,
        "qid": datum["qid"],
        "Question": question,
        "Tweet": tweet
    }


def remove_author_date(datum: dict):
    tweet = re.sub(r'\(@\D+\)\s\D+\d{1,2},\s\d{4}$', '', datum["Tweet"], flags=re.I)

    return {
        "Answer": datum["Answer"],
        "qid": datum["qid"],
        "Question": datum["Question"],
        "Tweet": tweet
    }


def remove_emojis(datum: dict):
    answer = datum["Answer"].encode('ascii', 'ignore').decode()
    tweet = datum["Tweet"].encode('ascii', 'ignore').decode()
    question = datum["Question"].encode('ascii', 'ignore').decode()

    return {
        "Answer": answer,
        "qid": datum["qid"],
        "Question": question,
        "Tweet": tweet
    }


def normalise_from_scoring(datum: dict):
    answer = normalize_answer(datum["Answer"])
    tweet = normalize_answer(datum["Tweet"])
    question = normalize_answer(datum["Question"])

    return {
        "Answer": answer,
        "qid": datum["qid"],
        "Question": question,
        "Tweet": tweet
    }


def normalise_datum(datum: dict):
    answer = ' '.join([str.strip(x) for x in normalise(datum["Answer"], verbose=False)])
    tweet = ' '.join([str.strip(x) for x in normalise(datum["Tweet"], verbose=False)])
    question = ' '.join([str.strip(x) for x in normalise(datum["Question"], verbose=False)])

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


def add_token_positions(tokenizer, encodings, answers, for_tf=False):
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

    # tensorflow will expect different names from pytorch
    if for_tf:
        start_name = 'start_logits'
        end_name = 'end_logits'
    else:
        start_name = 'start_positions'
        end_name = 'end_positions'

    encodings.update({start_name: start_positions, end_name: end_positions})


def do_filters(datum: dict):
    datum = lower_case_filter(datum)
    # datum = remove_author_date(datum)
    datum = do_abbrevs(datum)
    # datum = remove_author_date(datum)
    # datum = remove_groupings(datum) # seems to do nothing when using normalise_from_scoring
    # datum = remove_quotes(datum)    # seems to do nothing when using normalise_from_scoring
    datum = normalise_from_scoring(datum)
    # datum = remove_emojis(datum)
    # try:
    #     datum = normalise_datum(datum)
    # except BaseException:
    #     print('skipped normalising due to error, qid = ' + datum['qid'])
    #     pass
    datum = identify_start_and_end_positions(datum)
    return datum


def prepare_data(df: pd.DataFrame, for_tf=False, save_data=False, print_stats=False):
    df["Answer"] = df["Answer"].explode()
    train_data, val_data = train_test_split(df, test_size=0.2)
    x_train_before = train_data.to_dict('records')
    x_val_before = val_data.to_dict('records')

    x_train_filtered = [do_filters(datum) for datum in x_train_before if datum["qid"] not in qid_blacklist]
    x_val_filtered = [do_filters(datum) for datum in x_val_before if datum["qid"] not in qid_blacklist]

    non_quality_x_train = [datum for datum in x_train_filtered if datum["start_position"] == -1 or datum["end_position"] <= 0]
    non_quality_x_val = [datum for datum in x_val_filtered if datum["start_position"] == -1 or datum["end_position"] <= 0]
    quality_x_train = [datum for datum in x_train_filtered if datum["start_position"] >= 0 and datum["end_position"] > 0]
    quality_x_val = [datum for datum in x_val_filtered if datum["start_position"] >= 0 and datum["end_position"] > 0]

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
        with open('prep/x_val_before.json', 'w') as f_out:
            json.dump(x_val_before, f_out)
        with open('prep/x_train_filtered.json', 'w') as f_out:
            json.dump(x_train_filtered, f_out)
        with open('prep/quality_x_train.json', 'w') as f_out:
            json.dump(quality_x_train, f_out)
        with open('prep/non_quality_x_train.json', 'w') as f_out:
            json.dump(non_quality_x_train, f_out)
        with open('prep/quality_x_val.json', 'w') as f_out:
            json.dump(quality_x_val, f_out)
        with open('prep/non_quality_x_val.json', 'w') as f_out:
            json.dump(non_quality_x_val, f_out)

    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    train_encodings = tokenizer(
        [q["tweet"] for q in quality_x_train],
        [q["question"] for q in quality_x_train],
        padding='longest',
    )

    val_encodings = tokenizer(
        [q["tweet"] for q in quality_x_val],
        [q["question"] for q in quality_x_val],
        padding='longest',
    )

    add_token_positions(tokenizer, train_encodings, quality_x_train, for_tf)
    add_token_positions(tokenizer, val_encodings, quality_x_val, for_tf)

    return train_encodings, val_encodings
