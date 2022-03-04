import shutil
import os
from tqa_training_lib.data_extraction_lib import extract_data
from tqa_training_lib.data_preparation_lib import prepare_data
from tqa_training_lib.model_scoring_lib import score_model
from tqa_training_lib.trainers.tf_tweetqa_trainer import TFTweetQATrainer
from tqa_training_lib.trainers.tweetqa_training_args import TweetQATrainingArgs

model_out_path = 'model_out/'
# log_out_path = model_out_path + 'logs'

if os.path.exists(model_out_path):
    shutil.rmtree(model_out_path)

os.mkdir(model_out_path)

df = extract_data('https://raw.githubusercontent.com/sweng480-team23/tweet-qa-data/main/train.json')
train_encodings, val_encodings = prepare_data(df, save_data=True, print_stats=True, for_tf=True)

args = TweetQATrainingArgs(
    epochs=2,
    learning_rate=2.9e-5,
    batch_size=8,
    base_model='bert-large-uncased-whole-word-masking-finetuned-squad',
    model_output_path=model_out_path,
    use_cuda=True
)

print('---------------------- TRAINING ----------------------')

trainer = TFTweetQATrainer()
trainer.train(train_encodings, val_encodings, args)

print('---------------------- SCORING ----------------------')

score_model(model_out_path, save_gold_user_files=True, print_scores=True, use_tf=True)
