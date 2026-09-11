import os
from typing import Callable

from template import (
    OPENAI_MODEL,
    count_tokens,
    estimate_cost,
    retry_with_backoff,
)


def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    if get_input is None:
        get_input = input
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history, num_turns, total_tokens, total_cost = [], 0, 0, 0.0
    while True:
        if max_turns is not None and num_turns >= max_turns:
            break
        user_msg = get_input()
        if user_msg.strip().lower() in ("quit", "exit"):
            break
        messages = ([{"role": "system", "content": persona}]
                    + history + [{"role": "user", "content": user_msg}])
        stream = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, stream=True,
            )
        )
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
        print()
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]
        num_turns += 1
        total_tokens += count_tokens(user_msg) + count_tokens(reply)
        total_cost += estimate_cost(user_msg, reply)["total_cost"]
    return {"num_turns": num_turns, "total_tokens": total_tokens,
            "total_cost": total_cost, "history": history}