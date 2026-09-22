import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from image_captioning.data import Flickr8kDatasetBuilder, PickleDatasetRepository
from image_captioning.preprocessing import CaptionCleaner


class CaptioningCoreTests(unittest.TestCase):
    def test_caption_cleaner_adds_sequence_tokens(self):
        cleaner = CaptionCleaner()
        self.assertEqual(cleaner.clean("A dog, runs!"), "startseq dog runs endseq")

    def test_flickr8k_builder_splits_records_deterministically(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            captions_file = root / "captions.txt"
            lines = [f"image_{index}.jpg|{index}|caption {index} {caption}\n" for index in range(5) for caption in range(5)]
            captions_file.write_text("".join(lines), encoding="utf-8")

            builder = Flickr8kDatasetBuilder(root / "Images", train_ratio=0.8, seed=42)
            train, valid = builder.build(captions_file)
            self.assertEqual(len(train), 4)
            self.assertEqual(len(valid), 1)

            output = root / "processed" / "train.pkl"
            PickleDatasetRepository.save(train, output)
            self.assertEqual(PickleDatasetRepository.load(output), train)


if __name__ == "__main__":
    unittest.main()