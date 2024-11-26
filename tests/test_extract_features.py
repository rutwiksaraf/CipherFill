import pandas as pd
from unredactor import extract_features


def test_extract_features():
    df = pd.DataFrame({
        'name': ['John Doe', 'Jane Smith'],
        'context': ["The name is ██████████ and belongs to the person.",
                    "Here is ██████████ with some description."]
    })
    
    features = extract_features(df)
    assert not features.empty
    assert 'unigrams' in features.columns
    assert 'bigrams' in features.columns
    assert 'trigrams' in features.columns
    assert 'previous_word' in features.columns
    assert 'next_word' in features.columns