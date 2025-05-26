import os
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from utils import load_vector_db
import streamlit as st

# 🔐 Set your Groq API key securely
os.environ["GROQ_API_KEY"] = "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"

# 🔗 Initialize the ChatGroq LLM
llm = ChatGroq(
    model="llama3-8b-8192"  # Options: llama3-8b-8192, mixtral-8x7b-32768, gemma-7b-it
)

# 🧠 Load FAISS vector database
db = load_vector_db()

# 🌐 Streamlit UI
st.title("📄 PDF Chatbot 🤖")
query = st.text_input("Ask something from your PDF:")

if query:
    docs = db.similarity_search(query)
    if docs:
        context = docs[0].page_content
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        response = llm.invoke(prompt)

        st.write("### 🤖 Response:")
        st.write(response.content)
    else:
        st.warning("No relevant documents found for your query.")
