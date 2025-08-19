import streamlit as st
from chatbot import BaristaBot

# inisialisasi chatbot
if "bot" not in st.session_state:
    st.session_state.bot = BaristaBot()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("☕ BaristaBot Cafe")

# tampilkan riwayat chat
for chat in st.session_state.chat_history:
    role, msg = chat
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)

# input user
if prompt := st.chat_input("Ketik pesanmu..."):
    st.session_state.chat_history.append(("user", prompt))

    response = st.session_state.bot.handle_input(prompt)
    st.session_state.chat_history.append(("assistant", response))
    st.experimental_rerun()
