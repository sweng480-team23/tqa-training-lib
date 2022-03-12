import shutil
from statistics import mode
from tqa_training_lib.data_extraction_lib import extract_data
from tqa_training_lib.data_preparation_lib import prepare_data
from tqa_training_lib.model_scoring_lib import score_model
from tqa_training_lib.trainers.torch_tweetqa_trainer import TorchTweetQATrainer
import os

from tqa_training_lib.trainers.tweetqa_training_args import TweetQATrainingArgs

model_out_path = 'model_out/'
log_out_path = model_out_path + 'logs'

if os.path.exists(model_out_path):
    shutil.rmtree(model_out_path)

os.mkdir(model_out_path)

df = extract_data('https://raw.githubusercontent.com/sweng480-team23/tweet-qa-data/main/train.json')
train_encodings, val_encodings = prepare_data(df)

print('---------------------- TRAINING ----------------------')

args = TweetQATrainingArgs(
    epochs=2,
    learning_rate=2.9e-5,
    batch_size=8,
    base_model='bert-large-uncased-whole-word-masking-finetuned-squad',
    model_output_path=model_out_path,
    use_cuda=True
)

# f = open('model_out/args.json', 'w')
# json_obj = jsonpickle.encode(args)
# f.write(json_obj)

trainer = TorchTweetQATrainer()
trainer.train(train_encodings, val_encodings, args)

print('---------------------- SCORING ----------------------')

score_model(model_out_path)
