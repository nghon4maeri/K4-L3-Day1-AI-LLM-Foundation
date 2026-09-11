from template import PRICING_PER_1K_TOKENS, call_openai, call_openai_mini


def compare_models(prompt: str) -> dict:
    gpt4o_text, gpt4o_latency = call_openai(prompt)
    mini_text, mini_latency = call_openai_mini(prompt)
    cost = (len(gpt4o_text.split()) / 0.75) / 1000 \
           * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    return {
        "gpt4o_response": gpt4o_text,
        "mini_response": mini_text,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": cost,
    }