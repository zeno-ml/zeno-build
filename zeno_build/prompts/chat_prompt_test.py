"""Tests of the ChatPrompt class."""

from zeno_build.prompts.chat_prompt import ChatMessages, ChatTurn

example_prompt = ChatMessages(
    messages=[
        ChatTurn(
            role="system",
            content="You are a chatbot.",
        ),
        ChatTurn(role="system", content="{{context}}"),
        ChatTurn(role="user", content="{{source}}"),
    ]
)


def test_openai_chat_completion_messages():
    """Test generation of an OpenAI ChatCompletion messages format."""
    expected_messages = [
        {"role": "system", "content": "You are a chatbot."},
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "goodbye"},
    ]
    actual_messages = example_prompt.to_openai_chat_completion_messages(
        {"context": "hello", "source": "goodbye"}
    )

    assert expected_messages == actual_messages


def test_text_prompt():
    """Test generation of a regular textual format."""
    expected_text = (
        "System: You are a chatbot.\n\n"
        "System: hello\n\n"
        "User: goodbye\n\n"
        "System: "
    )
    actual_text = example_prompt.to_text_prompt(
        {"context": "hello", "source": "goodbye"}
    )

    assert expected_text == actual_text


def test_text_prompt_with_names():
    """Test generation of a regular textual format w/ names."""
    expected_text = (
        "Me: You are a chatbot.\n\n" "Me: hello\n\n" "You: goodbye\n\n" "Me: "
    )
    actual_text = example_prompt.to_text_prompt(
        variables={"context": "hello", "source": "goodbye"},
        system_name="Me",
        user_name="You",
    )

    assert expected_text == actual_text
