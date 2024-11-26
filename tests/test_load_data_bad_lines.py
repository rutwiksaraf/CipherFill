from unredactor import load_data


def test_load_data_bad_lines(tmp_path):
    test_file = tmp_path / "test_data.tsv"
    test_file.write_text("training\tJohn Doe\tThis is the context.\nbad\tdata\nvalidation\tJane Smith\tAnother context.")
    
    df = load_data(test_file)
    assert len(df) == 2  # Expect only valid rows
    assert df.iloc[0]['name'] == "John Doe"
    assert df.iloc[1]['name'] == "Jane Smith"
