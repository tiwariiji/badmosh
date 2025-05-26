from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
