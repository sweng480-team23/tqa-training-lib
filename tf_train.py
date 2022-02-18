import shutil
from statistics import mode
from tqa_training_lib.data_extraction_lib import extract_data
from tqa_training_lib.data_preparation_lib import prepare_data
from tqa_training_lib.model_scoring_lib import score_model
import os

from tqa_training_lib.tf_model_training_lib import do_train

model_out_path = 'model_out/'
log_out_path = model_out_path + 'logs'

if os.path.exists(model_out_path):
    shutil.rmtree(model_out_path)

os.mkdir(model_out_path)

df = extract_data('https://raw.githubusercontent.com/sweng480-team23/tweet-qa-data/main/train.json')
df = df.head(900)       # for some reason scoring breaks if we train on a larger set? under investigation
train_encodings, val_encodings = prepare_data(df, save_data=True, print_stats=True, for_tf=True)

print('---------------------- TRAINING ----------------------')

do_train(train_encodings, val_encodings)

print('---------------------- SCORING ----------------------')

score_model(model_out_path, save_gold_user_files=True, print_scores=True, use_tf=True)
