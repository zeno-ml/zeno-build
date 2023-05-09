"""Tests of modeling."""

from tasks.chatbot.modeling import build_examples_from_sequence
from zeno_build.prompts.chat_prompt import ChatMessages, ChatTurn


def test_build_examples_from_sequence():
    """Test build_examples_from_sequence."""
    expected_examples = [
        ChatMessages(
            messages=[
                ChatTurn(role="user", content="hello"),
                ChatTurn(role="assistant", content="how are you"),
            ]
        ),
        ChatMessages(
            messages=[
                ChatTurn(role="assistant", content="hello"),
                ChatTurn(role="user", content="how are you"),
                ChatTurn(role="assistant", content="goodbye"),
            ]
        ),
    ]
    actual_examples = list(
        build_examples_from_sequence(["hello", "how are you", "goodbye"])
    )

    assert expected_examples == actual_examples
