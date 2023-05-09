"""Tests of the ChatPrompt class."""

from zeno_build.prompts.chat_prompt import ChatMessages, ChatTurn

example_prompt = ChatMessages(
    messages=[
        ChatTurn(
            role="system",
            content="You are a chatbot.",
        ),
    ]
)


def test_openai_chat_completion_messages():
    """Test generation of an OpenAI ChatCompletion messages format."""
    expected_messages = [
        {"role": "system", "content": "You are a chatbot."},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "goodbye"},
    ]
    actual_messages = example_prompt.to_openai_chat_completion_messages(
        ChatMessages(
            messages=[
                ChatTurn(role="assistant", content="hello"),
                ChatTurn(role="user", content="goodbye"),
            ]
        ),
    )

    assert expected_messages == actual_messages


def test_text_prompt():
    """Test generation of a regular textual format."""
    expected_text = (
        "system: You are a chatbot.\n\n"
        "assistant: hello\n\n"
        "user: goodbye\n\n"
        "assistant:"
    )
    actual_text = example_prompt.to_text_prompt(
        ChatMessages(
            messages=[
                ChatTurn(role="assistant", content="hello"),
                ChatTurn(role="user", content="goodbye"),
            ]
        ),
        name_replacements={},
    )

    assert expected_text == actual_text


def test_text_prompt_with_names():
    """Test generation of a regular textual format w/ names."""
    expected_text = (
        "Me: You are a chatbot.\n\n" "Me: hello\n\n" "You: goodbye\n\n" "Me:"
    )
    actual_text = example_prompt.to_text_prompt(
        ChatMessages(
            messages=[
                ChatTurn(role="assistant", content="hello"),
                ChatTurn(role="user", content="goodbye"),
            ]
        ),
        name_replacements={"system": "Me", "assistant": "Me", "user": "You"},
    )

    assert expected_text == actual_text
