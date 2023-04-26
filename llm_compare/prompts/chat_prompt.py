"""Specify the prompts you want to use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llm_compare.prompts.prompt_utils import replace_variables


@dataclass
class ChatTurn:
    """A single turn for a chat prompt.

    Attributes:
        role: The role of the thing making the utterance.
        content: The utterance itself.
    """

    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True)
class ChatMessages:
    """A set of chat messages.

    Attributes:
        messages: A set of messages for the chat turn
    """

    messages: list[ChatTurn]

    def to_openai_chat_completion_messages(
        self,
        variables: dict[str, str],
    ) -> list[dict[str, str]]:
        """Build an OpenAI ChatCompletion message.

        Args:
            variables: The variables to be replaced in the prompt template.

        Returns:
            A list of dictionaries that can be consumed by the OpenAI API.
        """
        return [
            {
                "role": x.role,
                "content": replace_variables(x.content, variables),
            }
            for x in self.messages
        ]

    def to_text_prompt(
        self,
        variables: dict[str, str],
        system_name: str = "System",
        user_name: str = "User",
    ) -> str:
        """Create a normal textual prompt of a chat history.

        Args:
            variables: The variables to be replaced in the prompt template.
            system_name: The name of the system.
            user_name: The name of the user.

        Returns:
            str: _description_
        """
        return (
            "\n".join(
                [
                    system_name
                    if x.role == "system"
                    else user_name + ": " + replace_variables(x.content, variables)
                    for x in self.messages
                ]
            )
            + f"{system_name}: "
        )
