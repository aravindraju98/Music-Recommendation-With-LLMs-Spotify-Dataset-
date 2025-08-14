import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
import json
from typing import Callable


SYSTEM_PROMPT = (
    "You are a helpful music assistant. "
    "Ask the user to list several songs they like, including artist names if possible. "
    "After they reply, you will call a tool to search for the best matching track IDs from a local CSV. "
    "If needed, ask clarifying questions to disambiguate songs with common names. "
)


def get_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")
    return OpenAI(api_key=api_key)


def call_llm(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
    """Simple LLM chat completion wrapper returning assistant text."""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""


def _tool_specs():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_tracks",
                "description": "Fuzzy search the local CSV to resolve user-provided song text to track IDs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The user-provided song(s) and optional artists."},
                        "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                        "score_cutoff": {"type": "integer", "default": 70, "minimum": 0, "maximum": 100},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recommend_songs",
                "description": "Recommend similar tracks given resolved seed track IDs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "track_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Resolved track IDs to seed the recommender.",
                        },
                        "top_n": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                    },
                    "required": ["track_ids"],
                },
            },
        },
    ]


def chat_with_tools(
    messages: List[Dict[str, str]],
    tool_handlers: Dict[str, Callable[[Dict], Dict]],
    model: str = "gpt-4o-mini",
    max_tool_rounds: int = 3,
):
    """Run a chat with tool-calling. Returns (assistant_text, tool_outputs).

    tool_handlers: mapping function_name -> callable(args_dict) -> result_dict
    """
    client = get_openai_client()
    tool_outputs: Dict[str, Dict] = {}

    running_messages = list(messages)
    for _ in range(max_tool_rounds):
        resp = client.chat.completions.create(
            model=model,
            messages=running_messages,
            tools=_tool_specs(),
            temperature=0.4,
        )
        message = resp.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            # Final assistant message
            content = message.content or ""
            return content, tool_outputs

        # Append the assistant message that requested tools
        running_messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [
            {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tool_calls
        ]})

        # Execute tool calls and append tool results
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            handler = tool_handlers.get(name)
            if handler is None:
                result = {"error": f"No handler for tool {name}"}
            else:
                try:
                    result = handler(args) or {}
                except Exception as e:
                    result = {"error": str(e)}
            tool_outputs[name] = result
            running_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(result),
            })

    # Safety: if we exit due to max rounds, ask the model to finalize
    final = client.chat.completions.create(
        model=model,
        messages=running_messages,
        temperature=0.4,
    )
    return final.choices[0].message.content or "", tool_outputs


