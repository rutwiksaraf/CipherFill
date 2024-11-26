from unredactor import get_next_word


def test_get_next_word():
    context = "The quick ██████████ fox jumps."
    result = get_next_word(context)
    assert result == "fox"