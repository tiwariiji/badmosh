from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Step 1: Load PDF
loader = PyPDFLoader("DSEU_Admission Brochure 2025 _final_2152025.pdf")  # PDF file name
pages = loader.load()

# Step 2: Split text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(pages)

# Step 3: Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

# Step 4: Save locally
db.save_local("vectorstore")  # Folder created automatically

print("✅ Vector store created successfully.")
