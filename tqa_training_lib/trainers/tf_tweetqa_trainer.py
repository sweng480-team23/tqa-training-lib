import gc
from transformers import TFBertForQuestionAnswering
import tensorflow as tf

from tqa_training_lib.trainers.tweetqa_training_args import TweetQATrainingArgs
from tqa_training_lib.trainers.tweetqa_trainer import TweetQATrainer


class TFTweetQATrainer(TweetQATrainer):
    def train(self, train_encodings, val_encodings, args):
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

        train_dataset = tf.data.Dataset.from_tensor_slices((
            {key: train_encodings[key] for key in ['input_ids', 'attention_mask']},
            {key: train_encodings[key] for key in ['start_logits', 'end_logits']}
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            {key: val_encodings[key] for key in ['input_ids', 'attention_mask']},
            {key: val_encodings[key] for key in ['start_logits', 'end_logits']}
        ))

        model = TFBertForQuestionAnswering.from_pretrained(args.base_model)

        # Keras will expect a tuple when dealing with labels

        # Keras will assign a separate loss for each output and add them together. So we'll just use the standard CE loss
        # instead of using the built-in model.compute_loss, which expects a dict of outputs and averages the two terms.
        # Note that this means the loss will be 2x of when using TFTrainer since we're adding instead of averaging them.
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.return_dict = False  # if using 🤗 Transformers >3.02, make sure outputs are tuples

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        model.compile(optimizer=optimizer, loss=loss)   # can also use any keras loss fn
        model.fit(train_dataset.batch(args.batch_size), epochs=args.epochs, batch_size=args.batch_size)

        model.save_pretrained(args.model_output_path)
        del model, loss, optimizer, val_dataset, train_dataset
        gc.collect()
