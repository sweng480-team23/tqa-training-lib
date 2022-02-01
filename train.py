import shutil
from statistics import mode
from tqa_training_lib.data_extraction_lib import extract_data
from tqa_training_lib.data_preparation_lib import prepare_data
from tqa_training_lib.model_scoring_lib import score_model
from tqa_training_lib.model_training_lib import do_train
from transformers import TrainingArguments
import jsonpickle
import os

model_out_path = 'model_out/'
log_out_path = model_out_path + 'logs'

if os.path.exists(model_out_path):
    shutil.rmtree(model_out_path)

os.mkdir(model_out_path)

df = extract_data('https://raw.githubusercontent.com/sweng480-team23/tweet-qa-data/main/train.json')
train_encodings, val_encodings = prepare_data(df)

print('---------------------- TRAINING ----------------------')

training_args = TrainingArguments(
    output_dir=model_out_path,
    num_train_epochs=2,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    learning_rate=4.5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=log_out_path,
    logging_steps=500,
    save_strategy="steps",
    save_steps=500
)

f = open('model_out/args.json', 'w')
json_obj = jsonpickle.encode(training_args)
f.write(json_obj)

do_train(
    train_encodings,
    val_encodings,
    use_cuda=True,
    training_args=training_args
)

print('---------------------- SCORING ----------------------')

score_model(model_out_path)
