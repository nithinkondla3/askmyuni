import streamlit as st
from rag_chain import ask

st.set_page_config(page_title="AskMyUni", page_icon="🎓")
st.title("AskMyUni 🎓")
st.caption("Ask questions about your RMIT course handbook")
st.warning("⚠️ Always verify important info with RMIT official sources.")

# Initialize history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_pages" not in st.session_state:
    st.session_state.last_pages = []

# Clear button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.last_pages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("pages"):
            st.caption(f"📄 Sources: PDF pages {msg['pages']}")

# Sidebar
with st.sidebar:
    st.markdown("### 📄 Sources Used")
    if st.session_state.last_pages:
        for page in st.session_state.last_pages:
            st.markdown(f"**RMIT Handbook** — *Page {page}*")
    else:
        st.caption("Ask a question to see sources!")

# Handle new input
user_input = st.chat_input("Type your question here...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Searching handbook..."):
            answer, pages = ask(user_input)
        st.write(answer)
        if pages:
            st.caption(f"📄 Sources: PDF pages {pages}")

    st.session_state.last_pages = pages
    st.session_state.messages.append({"role": "assistant", "content": answer, "pages": pages})