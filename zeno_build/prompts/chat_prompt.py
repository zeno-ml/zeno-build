"""Specify the prompts you want to use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ChatTurn:
    """A single turn for a chat prompt.

    Attributes:
        role: The role of the thing making the utterance.
        content: The utterance itself.
    """

    role: Literal["system", "assistant", "user"]
    content: str


@dataclass(frozen=True)
class ChatMessages:
    """A set of chat messages.

    Attributes:
        messages: A set of messages for the chat turn
    """

    messages: list[ChatTurn]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ChatMessages:
        """Create a chat messages object from a dictionary."""
        return ChatMessages(
            messages=[
                ChatTurn(role=x["role"], content=x["content"]) for x in data["messages"]
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert a chat messages object to a dictionary."""
        return {
            "messages": [
                {
                    "role": x.role,
                    "content": x.content,
                }
                for x in self.messages
            ]
        }

    def to_openai_chat_completion_messages(
        self,
        full_context: ChatMessages,
    ) -> list[dict[str, str]]:
        """Build an OpenAI ChatCompletion message.

        Args:
            variables: The variables to be replaced in the prompt template.

        Returns:
            A list of dictionaries that can be consumed by the OpenAI API.
        """
        messages = [
            {
                "role": x.role,
                "content": x.content,
            }
            for x in self.messages + full_context.messages
        ]
        return messages

    def to_text_prompt(
        self,
        full_context: ChatMessages,
        name_replacements: dict[str, str],
    ) -> str:
        """Create a normal textual prompt of a chat history.

        Args:
            variables: The variables to be replaced in the prompt template.
            system_name: The name of the system.
            user_name: The name of the user.

        Returns:
            str: _description_
        """
        messages = [
            f"{name_replacements.get(x.role, x.role)}: {x.content}"
            for x in self.messages + full_context.messages
        ]
        assistant_name = name_replacements.get("assistant", "assistant")
        messages += [f"{assistant_name}:"]
        return "\n\n".join(messages)

    def limit_length(self, context_length: int) -> ChatMessages:
        """Limit the length of the chat history.

        Args:
            context_length: The maximum length of the context, or 0 or less for no
                limit.

        Returns:
            The chat messages with the context length limited.
        """
        return (
            ChatMessages(messages=self.messages[-context_length:])
            if context_length >= 0
            else self
        )
