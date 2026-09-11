"""
Chatbot UI — Streamlit, dùng lại logic từ template.py.
Chạy:  streamlit run chat_app.py
"""

import os

import streamlit as st

from template import (
    OPENAI_MINI_MODEL,
    OPENAI_MODEL,
    count_tokens,
    estimate_cost,
    retry_with_backoff,
)

DEFAULT_PERSONA = (
    "Bạn là trợ giảng thân thiện của khóa AI, "
    "trả lời ngắn gọn bằng tiếng Việt."
)


def get_client():
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def stream_reply(messages, model):
    """Generator stream từng chunk — dùng cho st.write_stream."""
    client = get_client()
    stream = retry_with_backoff(
        lambda: client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("num_turns", 0)
    st.session_state.setdefault("total_tokens", 0)
    st.session_state.setdefault("total_cost", 0.0)


def main():
    st.set_page_config(page_title="K4 Chatbot", page_icon="🤖")
    st.title("🤖 Trợ lý AI — K4 Day 1")

    init_state()

    with st.sidebar:
        st.header("Cấu hình")
        persona = st.text_area("Persona (system prompt)", DEFAULT_PERSONA, height=120)
        model_name = st.selectbox(
            "Model",
            ["gpt-4o (lớn)", "gpt-4o-mini (nhỏ)"],
            index=0,
        )
        model = OPENAI_MODEL if model_name.startswith("gpt-4o (lớn") else OPENAI_MINI_MODEL
        st.caption(f"Đang dùng: {model}")
        clear = st.button("Xóa hội thoại")

        st.divider()
        st.metric("Số lượt", st.session_state["num_turns"])
        st.metric("Tổng token", st.session_state["total_tokens"])
        st.metric("Chi phí ước tính", f"${st.session_state['total_cost']:.6f}")

    if clear:
        st.session_state["messages"] = []
        st.session_state["num_turns"] = 0
        st.session_state["total_tokens"] = 0
        st.session_state["total_cost"] = 0.0

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_msg := st.chat_input("Nhập tin nhắn của bạn..."):
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("fw_3xxxx"):
            st.error("Chưa có OPENAI_API_KEY hợp lệ — kiểm tra file .env")
            st.stop()

        st.session_state["messages"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        messages = (
            [{"role": "system", "content": persona}]
            + st.session_state["messages"]
        )

        try:
            with st.chat_message("assistant"):
                reply = st.write_stream(stream_reply(messages, model))
        except Exception as exc:
            st.error(f"Lỗi gọi API: {exc}")
            st.stop()

        st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.session_state["num_turns"] += 1
        st.session_state["total_tokens"] += count_tokens(user_msg) + count_tokens(reply)
        st.session_state["total_cost"] += estimate_cost(user_msg, reply)["total_cost"]
        st.rerun()


if __name__ == "__main__":
    main()