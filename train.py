from tqa_training_lib.data_extraction_lib import extract_data
from tqa_training_lib.data_preparation_lib import prepare_data
from tqa_training_lib.model_scoring_lib import score_model
from tqa_training_lib.model_training_lib import do_train


df = extract_data('https://raw.githubusercontent.com/sweng480-team23/tweet-qa-data/main/train.json')
train_encodings, val_encodings = prepare_data(df)

do_train(
    train_encodings,
    val_encodings,
    'model_out/',
    'model_out/logs',
    use_cuda=True
)

score_model('model_out/')
