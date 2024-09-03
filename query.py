import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Carrega o modelo de embeddings da Hugging Face
embeddings_model = HuggingFaceEmbeddings(model_name='models/sentence-transformers_paraphrase-multilingual-mpnet-base-v2')

# Função para realizar a procura no índice FAISS com um vetor de consulta
def search_faiss_with_embed_query(db, query, embeddings_model='models/sentence-transformers_paraphrase-multilingual-mpnet-base-v2', top_k=2):
    # Gerar embeddings da consulta usando embed_query
    query_embedding = embeddings_model.embed_query(query)

    # Query_embedding para um array NumPy
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding)

    # Procura no índice FAISS
    D, I = db.index.search(query_embedding.reshape(1, -1), top_k)

    # Recuperar os contextos correspondentes aos resultados mais relevantes
    results = []
    for i in I[0]:
        doc_id = db.index_to_docstore_id.get(i)  
        if doc_id in db.docstore._dict:
            document = db.docstore._dict[doc_id]
            results.append(document)
        else:
            print(f"Documento com ID {doc_id} não encontrado no docstore.")

    return results if results else None  # Retorna a lista de documentos encontrados ou None se estiver vazia

# Exemplo de uso da função search_faiss_with_embed_query
if __name__ == "__main__":
    # Carregar o índice FAISS existente
    db_path = 'vector_DB/db_faiss_excel'
    db = FAISS.load_local(db_path, embeddings=embeddings_model, allow_dangerous_deserialization=True)

    # Exemplo de consulta
    query = "Quando é o efolio a de Sistemas Distribuidos?"
    results = search_faiss_with_embed_query(db, query, embeddings_model)

    if results:
        print("Resultados encontrados:")
        for result in results:
            print(result.page_content)
    else:
        print("Nenhum resultado encontrado.")
