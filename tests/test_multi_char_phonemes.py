import whisperx


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