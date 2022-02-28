class TrainingArgsSimple(object):
    batch_size: int
    learning_rate: float
    model_output_path: str
    base_model: str
    epochs: str

    def __init__(self, batch_size, learning_rate, model_output_path, base_model, epochs):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_output_path = model_output_path
        self.base_model = base_model
        self.epochs = epochs
