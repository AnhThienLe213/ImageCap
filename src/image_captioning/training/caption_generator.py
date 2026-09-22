"""Batch generation for the VGG16/LSTM pipeline."""

import numpy as np


class CaptionBatchGenerator:
    """Keras-compatible generator that converts captions into teacher-forcing pairs."""

    def __init__(self, keys, mapping, features, tokenizer, max_length, vocabulary_size, batch_size):
        self.keys = keys
        self.mapping = mapping
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size

    def __iter__(self):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.utils import to_categorical

        while True:
            image_batch, sequence_batch, target_batch = [], [], []
            image_count = 0
            for key in self.keys:
                for caption in self.mapping[key]:
                    sequence = self.tokenizer.texts_to_sequences([caption])[0]
                    for index in range(1, len(sequence)):
                        image_batch.append(self.features[key][0])
                        sequence_batch.append(pad_sequences([sequence[:index]], maxlen=self.max_length, padding="post")[0])
                        target_batch.append(to_categorical([sequence[index]], num_classes=self.vocabulary_size)[0])
                image_count += 1
                if image_count == self.batch_size:
                    yield {"image": np.asarray(image_batch), "text": np.asarray(sequence_batch)}, np.asarray(target_batch)
                    image_batch, sequence_batch, target_batch = [], [], []
                    image_count = 0