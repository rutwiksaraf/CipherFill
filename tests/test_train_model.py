import pandas as pd
from unredactor import train_model

def test_train_model():
    train_df = pd.DataFrame({
        'split': ['training', 'training'],
        'name': ['John Doe', 'Jane Smith'],
        'context': ["The name is ██████████ and belongs to the person.",
                    "Here is ██████████ with some description."]
    })
    val_df = pd.DataFrame({
        'split': ['validation', 'validation'],
        'name': ['John Doe', 'Jane Smith'],
        'context': ["The name is ██████████ and belongs to someone.",
                    "Here is ██████████ with a different description."]
    })
    
    model, vectorizer = train_model(train_df, val_df)
    assert model is not None
    assert vectorizer is not None
