import whisperx
import numpy as np


def test_multichar_tokenizer_available():
    model, metadata = whisperx.load_align_model(
        language_code="de",
        device="cpu",
        model_name="facebook/wav2vec2-xlsr-53-espeak-cv-ft",
    )

    tokenizer = metadata.get("tokenizer")

    assert tokenizer is not None

    tokens = tokenizer.tokenize("einst")

    assert tokens[:4] == ["aɪ", "n", "s", "t"]

    dummy_audio = np.zeros(16000, dtype=np.float32)

    alignments = whisperx.align(
        [{"text": "einst", "start": 0.0, "end": 1.0}],
        model,
        metadata,
        dummy_audio,
        "cpu",
        return_char_alignments=True
    )

    print(alignments)