from template import OPENAI_MODEL


def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)