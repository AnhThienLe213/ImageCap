"""VGG16 + LSTM captioning model."""


class VGG16LSTMCaptioner:
    """Encapsulates construction and inference for the original baseline model."""

    def __init__(self, vocabulary_size: int, max_length: int, embedding_size: int = 256) -> None:
        self.vocabulary_size = vocabulary_size
        self.max_length = max_length
        self.embedding_size = embedding_size
        self.model = None

    def build(self):
        from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, Input, LSTM
        from tensorflow.keras.models import Model

        image_input = Input(shape=(4096,), name="image")
        image_features = Dense(self.embedding_size, activation="relu")(Dropout(0.4)(image_input))
        text_input = Input(shape=(self.max_length,), name="text")
        text_features = LSTM(self.embedding_size)(Dropout(0.4)(Embedding(
            self.vocabulary_size, self.embedding_size, mask_zero=True
        )(text_input)))
        decoder = Dense(self.embedding_size, activation="relu")(Add()([image_features, text_features]))
        output = Dense(self.vocabulary_size, activation="softmax")(decoder)
        self.model = Model(inputs=[image_input, text_input], outputs=output)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
        return self.model

    def load(self, path: str):
        from tensorflow.keras.models import load_model

        self.model = load_model(path)
        return self.model

    def predict_next(self, image_features, sequence):
        if self.model is None:
            raise RuntimeError("Build or load the model before prediction")
        return self.model.predict([image_features, sequence], verbose=0)