from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# 🧠 Load PDF and split
loader = PyPDFLoader("DSEU_Admission Brochure 2025 _final_2152025.pdf")  # Replace with your actual PDF
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 🔗 Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🧱 Create vector DB
db = FAISS.from_documents(texts, embeddings)
db.save_local("faiss_index")  # Save in the same folder where app.py expects it
