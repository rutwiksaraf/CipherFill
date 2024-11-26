from unredactor import get_next_word


def test_get_next_word_no_next():
    context = "The quick ██████████"
    result = get_next_word(context)
    assert result == ""