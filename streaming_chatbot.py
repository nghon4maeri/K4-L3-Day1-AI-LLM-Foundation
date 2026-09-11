import os

from template import OPENAI_MODEL


def streaming_chatbot() -> None:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history = []
    while True:
        user_msg = input("Bạn: ")
        if user_msg.strip().lower() in ("quit", "exit"):
            break
        messages = history + [{"role": "user", "content": user_msg}]
        stream = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, stream=True,
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