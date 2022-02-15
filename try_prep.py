from tqa_training_lib.data_extraction_lib import extract_data
from tqa_training_lib.data_preparation_lib import prepare_data

df = extract_data('https://raw.githubusercontent.com/sweng480-team23/tweet-qa-data/main/train.json')
train_encodings, val_encodings = prepare_data(df, save_data=True, print_stats=True)
