from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

DB_FAISS_PATH_UAB = 'vector_DB/db_faiss_uab'

def create_vector_db_from_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    embeddings = HuggingFaceEmbeddings(model_name='models/sentence-transformers_paraphrase-multilingual-mpnet-base-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(documents, embeddings)
    
    db.save_local(DB_FAISS_PATH_UAB)

if __name__ == "__main__":
    create_vector_db_from_text('context/UAB.txt')
