"""Caption quality metrics."""


class CaptionMetrics:
    """Compute BLEU metrics while keeping metric dependencies optional."""

    def score(self, prediction: str, references: list[str]) -> dict[str, float]:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        predicted_tokens = prediction.lower().split()
        reference_tokens = [reference.lower().split() for reference in references]
        smooth = SmoothingFunction().method1
        weights = {
            "bleu1": (1.0, 0, 0, 0),
            "bleu2": (0.5, 0.5, 0, 0),
            "bleu3": (1 / 3, 1 / 3, 1 / 3, 0),
            "bleu4": (0.25, 0.25, 0.25, 0.25),
        }
        return {
            name: round(sentence_bleu(reference_tokens, predicted_tokens, weights=weight, smoothing_function=smooth), 4)
            for name, weight in weights.items()
        }