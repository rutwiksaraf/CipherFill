from unredactor import get_previous_word


def test_get_previous_word_no_previous():
    context = "██████████ fox jumps."
    result = get_previous_word(context)
    assert result == ""