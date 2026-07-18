from __future__ import annotations

import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL = os.getenv(
    "LOGICGUARD_NLI_MODEL",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
)


def main() -> None:
    print(f"Downloading tokenizer: {MODEL}")
    AutoTokenizer.from_pretrained(MODEL)
    print(f"Downloading model: {MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    print(f"Ready. Labels: {model.config.id2label}")


if __name__ == "__main__":
    main()
