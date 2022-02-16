from transformers import TFAutoModelForQuestionAnswering, TFTrainer, TFTrainingArguments
import tensorflow as tf


def do_train(train_encodings, val_encodings, use_cuda=False, training_args=None):
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {key: train_encodings[key] for key in ['input_ids', 'attention_mask']},
        {key: train_encodings[key] for key in ['start_positions', 'end_positions']}
    ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        {key: val_encodings[key] for key in ['input_ids', 'attention_mask']},
        {key: val_encodings[key] for key in ['start_positions', 'end_positions']}
    ))

    model = TFAutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    device = 'cpu'

    if use_cuda:
        device = 'cuda'

    print('using device: ' + device)
    model = model.to(device)

    # Keras will expect a tuple when dealing with labels
    train_dataset = train_dataset.map(lambda x, y: (x, (y['start_positions'], y['end_positions'])))

    # Keras will assign a separate loss for each output and add them together. So we'll just use the standard CE loss
    # instead of using the built-in model.compute_loss, which expects a dict of outputs and averages the two terms.
    # Note that this means the loss will be 2x of when using TFTrainer since we're adding instead of averaging them.
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # model.distilbert.return_dict = False # if using 🤗 Transformers >3.02, make sure outputs are tuples

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=loss)   # can also use any keras loss fn
    # model.fit(train_dataset.shuffle(1000).batch(12), epochs=3, batch_size=12)
    model.fit(train_dataset.batch(12), epochs=3, batch_size=12)

    # if training_args is None:
    #     training_args = TFTrainingArguments(
    #         output_dir='model_out/',
    #         num_train_epochs=2,
    #         per_device_train_batch_size=12,
    #         per_device_eval_batch_size=12,
    #         warmup_steps=500,
    #         weight_decay=0.01,
    #         logging_dir='model_out/log',
    #         logging_steps=500,
    #         save_strategy="steps",
    #         save_steps=500
    #     )

    # bert_model.train()

    # trainer = TFTrainer(
    #     model=bert_model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     compute_metrics=compute_metrics
    # )

    # trainer.train()
    # trainer.evaluate()
    # trainer.save_model(training_args.output_dir)
