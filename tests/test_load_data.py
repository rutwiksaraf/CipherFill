from unredactor import load_data


def test_load_data(tmp_path):
    test_file = tmp_path / "test_data.tsv"
    test_file.write_text("training\tJohn Doe\tThis is the context.\nvalidation\tJane Smith\tAnother context.")
    
    df = load_data(test_file)
    assert not df.empty
    assert list(df.columns) == ['split', 'name', 'context']
    assert len(df) == 2