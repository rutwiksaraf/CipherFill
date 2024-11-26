from unredactor import get_previous_word


def test_get_previous_word():
    context = "The quick ██████████ fox jumps."
    result = get_previous_word(context)
    assert result == "quick"