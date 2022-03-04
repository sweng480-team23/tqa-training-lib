from transformers import TrainingArguments


class TweetQATrainingArgs(object):
    batch_size: int
    learning_rate: float
    model_output_path: str
    base_model: str
    epochs: str
    use_cuda: bool

    def __init__(self, batch_size, learning_rate, model_output_path, base_model, epochs, use_cuda):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_output_path = model_output_path
        self.base_model = base_model
        self.epochs = epochs
        self.use_cuda = use_cuda

    def to_huggingface_trainer_arguments(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.model_output_path,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=500,
            save_strategy="steps",
            save_steps=500
        )
