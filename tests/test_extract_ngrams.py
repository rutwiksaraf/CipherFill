from unredactor import extract_ngrams


def test_extract_ngrams():
    text = "This is a ██████████ token for testing purposes."
    redacted_length = 10
    result = extract_ngrams(text, redacted_length)
    
    assert 'unigrams' in result
    assert 'bigrams' in result
    assert 'trigrams' in result
    assert isinstance(result['unigrams'], str)
    assert isinstance(result['bigrams'], str)
    assert isinstance(result['trigrams'], str)