import streamlit as st
st.set_page_config(page_title="Llama Chatbot", page_icon="🤖", layout="centered")
from llama_cpp import Llama
import os

# Model dosyasının yolu
MODEL_PATH = r"C:/Users/menesnas/Desktop/LLM AGENT/LLM/Llama2Chatbot/models/llama-2-7b-chat.Q2_K.gguf"

# Modeli sadece bir kez yükle (cache)
@st.cache_resource(show_spinner=True)
def load_llama_model():
    return Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=os.cpu_count() or 4)

llm = load_llama_model()

def get_llama_response(prompt: str) -> str:
    output = llm(
        prompt,
        max_tokens=256,
        stop=["</s>"],
        echo=False,
        temperature=0.7,
        top_p=0.95
    )
    return output["choices"][0]["text"].strip()

st.title("🤖 LLama Chatbot")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Mesajınızı yazın:", "")
    submitted = st.form_submit_button("Gönder")

if submitted and user_input.strip():
    st.session_state["chat_history"].append(("Siz", user_input))
    with st.spinner("Llama düşünüyor..."):
        response = get_llama_response(user_input)
    st.session_state["chat_history"].append(("Llama", response))

# Sohbet geçmişini göster
for sender, message in st.session_state["chat_history"]:
    if sender == "Siz":
        st.markdown(f"<div style='text-align: right; color: #1a73e8;'><b>{sender}:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; color: #34a853;'><b>{sender}:</b> {message}</div>", unsafe_allow_html=True) 