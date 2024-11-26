import pandas as pd
from unredactor import predict_and_generate_submission

def test_predict_and_generate_submission(tmp_path):
    test_file = tmp_path / "test_data.tsv"
    test_file.write_text("1\tThis is a ██████████ test.\n2\tAnother ██████████ example.", encoding='utf-8')

    
    output_file = tmp_path / "submission.tsv"
    
    # Mock model and vectorizer
    class MockModel:
        def predict(self, X):
            return ["MockName1", "MockName2"]
    
    class MockVectorizer:
        def transform(self, X):
            return X  # Dummy transformation
    
    model = MockModel()
    vectorizer = MockVectorizer()
    
    predict_and_generate_submission(model, vectorizer, test_file, output_file)
    
    # Verify output file
    submission_df = pd.read_csv(output_file, sep='\t')
    assert not submission_df.empty
    assert list(submission_df.columns) == ['id', 'name']
    assert len(submission_df) == 2
