"""Specify the prompts you want to use."""

from dataclasses import dataclass
from typing import Literal


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
    messages = list[ChatTurn]
    
    def to_openai_chat_completion_messages(self,
        source: str,
        context: str,
    ) -> list[dict[str, str]]:
        """Build an OpenAI ChatCompletion.

        Args:
            context: The context from the previous utterance.
            source: The current utterance from the user.

        Returns:
            A list of dictionaries that can be consumed by the OpenAI API.
        """
        return [
            {
                "role": x.role,
                "content": x.content.replace("{{context}}", context).replace("{{source}}", source),
            } for x in self.messages
        ]
    
    def to_text_prompt(
        self, 
        source: str, 
        context: str,  
        system_name: str = "System", 
        user_name: str = "User",
    ) -> str:
        return '\n'.join([
            system_name if x.role == "system" else user_name +
            ": " +
            x.content.replace("{{context}}", context).replace("{{source}}", source)
            for x in self.messages
        ]) + f"{system_name}: "    


prompt_messages: dict[str, ChatMessages] = {
    "standard": ChatMessages([
        ChatTurn(role="system", content="You are a chatbot tasked with making small-talk with people."),
        ChatTurn(role="system", content="{{context}}"),
        ChatTurn(role="user", content="{{source}}"),
    ]),
    "friendly": ChatMessages([
        ChatTurn(role="system", content="You are a kind and friendly chatbot tasked with making small-talk with people in a way that makes them feel pleasant."),
        ChatTurn(role="system", content="{{context}}"),
        ChatTurn(role="user", content="{{source}}"),
    ]),
    "polite": ChatMessages([
        ChatTurn(role="system", content="You are an exceedingly polite chatbot that speaks very formally and tries to not make any missteps in your responses."),
        ChatTurn(role="system", content="{{context}}"),
        ChatTurn(role="user", content="{{source}}"),
    ]),
    "cynical": ChatMessages([
        ChatTurn(role="system", content="You are a cynical chatbot that has a very dark view of the world and in general likes to point out any possible problems."),
        ChatTurn(role="system", content="{{context}}"),
        ChatTurn(role="user", content="{{source}}"),
    ])
}