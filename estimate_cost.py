from template import OPENAI_MODEL, PRICING_PER_1K_TOKENS, count_tokens


def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)
    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
    input_cost = input_tokens / 1000 * pricing["input"]
    output_cost = output_tokens / 1000 * pricing["output"]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }