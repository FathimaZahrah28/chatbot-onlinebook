import streamlit as st
from chatbot import chatbot_response

st.set_page_config(page_title="Online Book Chatbot", page_icon="📚")

st.title("📚 Online Book Chatbot")
st.write("Tanya apa saja tentang buku, rekomendasi, atau detail buku favoritmu!")

# inisialisasi history chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# tampilkan history chat
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input user
if prompt := st.chat_input("Ketik pertanyaanmu tentang buku..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # tampilkan pesan user
    with st.chat_message("user"):
        st.markdown(prompt)

    # dapatkan jawaban chatbot
    response = chatbot_response(prompt)

    # simpan jawaban bot ke history
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # tampilkan pesan bot
    with st.chat_message("assistant"):
        st.markdown(response)
