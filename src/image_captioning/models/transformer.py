"""Transformer captioner and early-exit heads."""

try:
    import torch.nn as nn
except ImportError:
    nn = None


class TransformerCaptioner:
    """Owns the Hugging Face vision-encoder/decoder model and its processors."""

    def __init__(self, encoder_name: str = "google/vit-base-patch16-224", decoder_name: str = "gpt2") -> None:
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.model = None
        self.tokenizer = None
        self.image_processor = None

    def load(self, checkpoint: str, device: str = "cpu"):
        from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

        self.model = VisionEncoderDecoderModel.from_pretrained(checkpoint).to(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.decoder_name)
        self.image_processor = ViTImageProcessor.from_pretrained(self.encoder_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        return self


class EarlyExitHead(nn.Module if nn is not None else object):
    """Projection head trained on an intermediate decoder hidden state."""

    def __init__(self, input_size: int, vocabulary_size: int) -> None:
        if nn is None:
            raise ImportError("EarlyExitHead requires PyTorch")
        super().__init__()
        self.module = nn.Linear(input_size, vocabulary_size)

    def forward(self, hidden_states):
        return self.module(hidden_states)